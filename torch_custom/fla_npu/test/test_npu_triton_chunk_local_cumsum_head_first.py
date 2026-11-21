#!/usr/bin/env python3
"""Regression test for ``head_first`` in Triton ``chunk_local_cumsum``.

The Triton kernel is token-major.  ``head_first=True`` used to be forwarded to
the kernel, which ignored it, so a ``[B, H, T]`` input was reinterpreted as
``[B, T, H]`` and silently produced wrong values.  The AscendC backend accepts
``[B, H, T]`` for the same flag, so both backends now agree.

Run from the repository root on an NPU host::

    python torch_custom/fla_npu/test/test_npu_triton_chunk_local_cumsum_head_first.py --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "fla/ops/triton/triton_core/cumsum.py").exists():
            return parent
    raise RuntimeError("repository root not found")


sys.path.insert(0, str(_repo_root()))

try:
    # Installed wheel: goes through the public package path.
    from fla_npu.ops.triton import chunk_local_cumsum  # noqa: E402
except Exception:  # noqa: BLE001 - a source checkout has no packaged OPP
    from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum  # noqa: E402


def prepare_chunk_indices(cu_seqlens: torch.Tensor,
                          chunk_size: int) -> torch.Tensor:
    """Canonical ``(seq, local_chunk)`` table."""

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks = [torch.arange((int(n) + chunk_size - 1) // chunk_size) for n in lengths.tolist()]
    indices = torch.cat(chunks)
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

ATOL = 1e-5


def _reference(g: torch.Tensor, chunk_size: int, *, reverse: bool = False,
               scale: float | None = None,
               cu_seqlens: list[int] | None = None) -> torch.Tensor:
    """Per-chunk cumulative sum of a head-first ``[B, H, T]`` tensor."""

    batch, heads, tokens = g.shape
    out = torch.empty_like(g, dtype=torch.float32)
    ranges = ([(0, tokens)] if cu_seqlens is None
              else list(zip(cu_seqlens, cu_seqlens[1:])))
    for b in range(batch):
        for h in range(heads):
            for begin, end in ranges:
                for start in range(begin, end, chunk_size):
                    stop = min(start + chunk_size, end)
                    block = g[b, h, start:stop].float()
                    if reverse:
                        block = torch.flip(torch.cumsum(torch.flip(block, [0]), 0), [0])
                    else:
                        block = torch.cumsum(block, 0)
                    if scale is not None:
                        block = block * scale
                    out[b, h, start:stop] = block
    return out


def check_case(device: str, batch: int, heads: int, tokens: int,
               chunk_size: int, *, reverse: bool = False,
               scale: float | None = None,
               cu_seqlens: list[int] | None = None) -> None:
    torch.manual_seed(500 + batch * 11 + heads * 7 + tokens + chunk_size)
    g = torch.randn(batch, heads, tokens, device=device, dtype=torch.float32)
    cu = None if cu_seqlens is None else torch.tensor(
        cu_seqlens, device=device, dtype=torch.int64)
    indices = None if cu is None else prepare_chunk_indices(cu, chunk_size)

    kwargs = dict(reverse=reverse, scale=scale, cu_seqlens=cu,
                  chunk_indices_out=None if indices is None else {str(chunk_size): indices},
                  output_dtype=torch.float32)

    actual = chunk_local_cumsum(g, chunk_size, head_first=True, **kwargs)
    expected = _reference(g, chunk_size, reverse=reverse, scale=scale,
                          cu_seqlens=cu_seqlens)
    assert actual.shape == expected.shape
    error = (actual - expected).abs().max().item()
    assert error < ATOL, (
        f"head_first mismatch B={batch} H={heads} T={tokens} C={chunk_size} "
        f"cu={cu_seqlens} reverse={reverse} scale={scale}: max={error:.3e}")

    # Token-major stays unchanged.
    token_first = chunk_local_cumsum(g.transpose(1, 2).contiguous(), chunk_size,
                                     head_first=False, **kwargs)
    error = (token_first - expected.transpose(1, 2)).abs().max().item()
    assert error < ATOL, f"token-first regression: max={error:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("SKIP: torch_npu is not available")
        return 0

    device = f"npu:{args.device}"
    torch.npu.set_device(int(args.device))

    cases = (
        dict(batch=2, heads=3, tokens=70, chunk_size=16),
        dict(batch=2, heads=3, tokens=70, chunk_size=16, reverse=True, scale=2.5),
        dict(batch=1, heads=4, tokens=70, chunk_size=16, cu_seqlens=[0, 17, 70]),
        dict(batch=1, heads=4, tokens=205, chunk_size=32, cu_seqlens=[0, 100, 205],
             reverse=True),
    )
    for case in cases:
        check_case(device, **case)
        print("PASS", case, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())