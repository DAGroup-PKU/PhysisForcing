#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Micro-benchmark: how much does ONE CoTracker3 call cost?

CoTracker dominates the per-step physics overhead, and it currently runs twice
per sample (low-level trajectory loss + high-level region mask). This script
times a single ``cotracker(video, grid_size=...)`` forward at several input
resolutions and frame counts so you can size the win from (a) de-duplicating the
two calls and (b) shrinking the input resolution (the loss is normalized by
H,W, so smaller input is numerically safe).

Run it in your venv on a GPU (content does not affect timing, so a random clip
is representative to within a few %):

    source .venv/bin/activate && export LD_LIBRARY_PATH=
    python -m tools.physics.bench_cotracker --frames 49 --grid 25 --res 256 192 128

To compare against an UPscaled 480p input (to see why "256 -> 480" is the wrong
direction), add 480 to --res.
"""

from __future__ import annotations

import argparse
import time

import torch

from cosmos_framework.model.vfm.physics.teachers import load_cotracker


@torch.no_grad()
def time_call(model, *, frames: int, h: int, w: int, grid: int, iters: int, warmup: int) -> float:
    """Return mean wall-clock ms for one cotracker() forward."""
    video = (torch.rand(1, frames, 3, h, w, device="cuda") * 255.0).float()
    for _ in range(warmup):
        model(video, grid_size=grid, grid_query_frame=0, backward_tracking=None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(video, grid_size=grid, grid_query_frame=0, backward_tracking=None)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="checkpoints/physics/co-tracker", help="vendored CoTracker hub repo")
    ap.add_argument("--frames", type=int, default=49, help="#pixel frames per clip (scales ~linearly)")
    ap.add_argument("--grid", type=int, default=25, help="track_grid_size (N_query = grid^2)")
    ap.add_argument("--res", type=int, nargs="+", default=[256, 192, 128], help="input HEIGHTS to test (16:9)")
    ap.add_argument("--samples-per-pack", type=int, default=20, help="for per-iter extrapolation")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    model = load_cotracker(args.repo, device="cuda")

    print(f"\nCoTracker3 offline | frames={args.frames} grid={args.grid} (N_query={args.grid**2})")
    print(f"{'height':>7} {'WxH':>11} {'ms/call':>9} {'x2 (cur)':>9} {'x2*S/iter':>11}")
    print("-" * 52)
    base = None
    for h in args.res:
        w = int(round(h * 16 / 9 / 2) * 2)  # 16:9, even width
        ms = time_call(model, frames=args.frames, h=h, w=w, grid=args.grid, iters=args.iters, warmup=args.warmup)
        if base is None:
            base = ms
        per_iter = ms * 2 * args.samples_per_pack / 1e3  # 2 calls/sample * S samples, in seconds
        print(f"{h:>7} {f'{w}x{h}':>11} {ms:>9.1f} {ms * 2:>9.1f} {per_iter:>10.1f}s")
    print(
        "\nx2  = current redundant calls/sample (low-level + mask)."
        f"\nx2*S/iter assumes S={args.samples_per_pack} samples packed per step on this rank."
        "\nDe-dup makes it x1; shrinking height cuts ms/call further (loss is H,W-normalized)."
    )


if __name__ == "__main__":
    main()
