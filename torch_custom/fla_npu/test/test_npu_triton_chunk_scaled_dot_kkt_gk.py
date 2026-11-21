#!/usr/bin/env python3
"""Regression test for the Triton ``chunk_scaled_dot_kkt_fwd`` gk path.

The gk kernels read ``k``/``beta``/``gk`` in token-major ``[B, T, H, *]``
layout while the public contract of ``k`` is head-major ``[B, H, T, K]``, and
they used to resolve the varlen chunk metadata inside the per-head loop.  Both
defects only show up with ``H > 1``; the second one additionally needs a
varlen chunk whose local index differs from its position in
``chunk_indices``.

Run from the repository root on an NPU host::

    python torch_custom/fla_npu/test/test_npu_triton_chunk_scaled_dot_kkt_gk.py --device 0

Only the lower-triangular entries the operator promises are compared; the
padding columns of a varlen tail chunk are intentionally left undefined.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "fla/ops/triton/triton_core/chunk_scaled_dot_kkt.py").exists():
            return parent
    raise RuntimeError("repository root not found")


sys.path.insert(0, str(_repo_root()))

try:
    # Installed wheel: goes through the public package path.
    from fla_npu.ops.triton import chunk_scaled_dot_kkt_fwd  # noqa: E402
except Exception:  # noqa: BLE001 - a source checkout has no packaged OPP
    from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import (  # noqa: E402
        chunk_scaled_dot_kkt_fwd,
    )


def prepare_chunk_indices(cu_seqlens: torch.Tensor,
                          chunk_size: int) -> torch.Tensor:
    """Canonical ``(seq, local_chunk)`` table."""

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks = [torch.arange((int(n) + chunk_size - 1) // chunk_size) for n in lengths.tolist()]
    indices = torch.cat(chunks)
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

ATOL = 1e-4


def _used_entries(batch: int, tokens: int, heads: int, chunk_size: int,
                  cu_seqlens: list[int] | None) -> torch.Tensor:
    """Mask of entries inside the chunk, strictly below the diagonal."""

    mask = torch.zeros(batch, tokens, heads, chunk_size, dtype=torch.bool)
    ranges = ([(0, tokens)] if cu_seqlens is None
              else list(zip(cu_seqlens, cu_seqlens[1:])))
    for begin, end in ranges:
        for start in range(begin, end, chunk_size):
            length = min(start + chunk_size, end) - start
            for row in range(length):
                mask[:, start + row, :, :row] = True
    return mask


def _call(device: str, k: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor | None, gk: torch.Tensor | None,
          chunk_size: int, cu_seqlens: torch.Tensor | None,
          chunk_indices: torch.Tensor | None) -> torch.Tensor:
    return chunk_scaled_dot_kkt_fwd(
        k=k,
        g=None if g is None else g.transpose(1, 2).contiguous(),
        gk=gk,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )


def check_case(device: str, heads: int, dim: int, chunk_size: int, tokens: int,
               *, batch: int = 1, cu_seqlens: list[int] | None = None) -> None:
    torch.manual_seed(1000 + heads * 17 + dim * 7 + chunk_size + tokens)
    k = torch.randn(batch, heads, tokens, dim, device=device, dtype=torch.float32)
    beta = torch.rand(batch, tokens, heads, device=device, dtype=torch.float32) + 0.5
    g = torch.randn(batch, heads, tokens, device=device, dtype=torch.float32) * 0.1

    cu = None if cu_seqlens is None else torch.tensor(
        cu_seqlens, device=device, dtype=torch.int64)
    indices = None if cu is None else prepare_chunk_indices(cu, chunk_size)

    reference = _call(device, k, beta, g, None, chunk_size, cu, indices)

    # Expanding the token gate over K makes the per-K gate path mathematically
    # identical to the scalar-gate path, so it doubles as an exact reference.
    gk = g.transpose(1, 2)[:, :, :, None].expand(
        batch, tokens, heads, dim).contiguous()
    actual = _call(device, k, beta, g, gk, chunk_size, cu, indices)

    mask = _used_entries(batch, tokens, heads, chunk_size, cu_seqlens)
    diff = (actual - reference).abs()[mask]
    assert diff.max().item() < ATOL, (
        f"gk path mismatch H={heads} K={dim} C={chunk_size} T={tokens} "
        f"B={batch} cu={cu_seqlens}: max={diff.max().item():.3e}")

    # g=None is the documented default; it must behave like an all-zero gate.
    if gk is not None and cu_seqlens is None and heads == 1:
        zeros = torch.zeros_like(g)
        without_gate = _call(device, k, beta, None, None, chunk_size, cu, indices)
        zero_gate = _call(device, k, beta, zeros, None, chunk_size, cu, indices)
        assert (without_gate - zero_gate).abs().max().item() < ATOL, (
            "g=None diverges from an all-zero gate")


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
        dict(heads=1, dim=16, chunk_size=16, tokens=64),
        dict(heads=2, dim=64, chunk_size=64, tokens=64),
        dict(heads=4, dim=128, chunk_size=128, tokens=64),
        dict(heads=2, dim=64, chunk_size=64, tokens=128, cu_seqlens=[0, 64, 128]),
        dict(heads=2, dim=64, chunk_size=64, tokens=200, cu_seqlens=[0, 70, 200]),
        dict(heads=2, dim=64, chunk_size=64, tokens=70, cu_seqlens=[0, 17, 70]),
        dict(heads=2, dim=64, chunk_size=32, tokens=205, cu_seqlens=[0, 100, 205]),
        dict(heads=2, dim=64, chunk_size=64, tokens=70, batch=2),
    )
    for case in cases:
        check_case(device, **case)
        print("PASS", case, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())