"""
Time VGGT's aggregator alone vs aggregator+heads on the A6000, across
sequence lengths S. This gives us the cost-table entries for VGGT's
split points (encoder-only vs full model), which is what the scheduler
actually decides between.
"""
import sys
import time
from pathlib import Path

import torch

VGGT_ROOT = Path("/home/myid/zs89458/Documents/vggt")
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def time_fn(fn, warm: int = 2, runs: int = 5) -> tuple[float, float, float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return min(ts), sum(ts) / len(ts), max(ts)


def main() -> None:
    device = "cuda"
    print(f"loading VGGT-1B on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    # Stack example kitchen images to make longer sequences; reuse same frames.
    paths = sorted((VGGT_ROOT / "examples" / "kitchen" / "images").glob("*"))[:1]
    base, _ = load_and_preprocess_images_square([str(paths[0])], target_size=518)
    base = base.to(device)  # (1, 3, 518, 518)

    for S in (1, 4, 8, 16):
        imgs = base.repeat(S, 1, 1, 1).unsqueeze(0)  # (1, S, 3, 518, 518)
        print(f"\n--- S = {S} frames, images: {tuple(imgs.shape)} ---")

        # Aggregator-only
        def aggregator_only():
            with torch.inference_mode():
                model.aggregator(imgs)

        # Full model (aggregator + all heads)
        def full_forward():
            with torch.inference_mode():
                model(imgs)

        agg_min, agg_avg, agg_max = time_fn(aggregator_only)
        full_min, full_avg, full_max = time_fn(full_forward)

        print(f"  aggregator only : {1000*agg_avg:.1f} ms  (min {1000*agg_min:.1f}, max {1000*agg_max:.1f})")
        print(f"  full (+ heads)  : {1000*full_avg:.1f} ms  (min {1000*full_min:.1f}, max {1000*full_max:.1f})")
        heads_ms = 1000 * (full_avg - agg_avg)
        print(f"  heads contribute: ~{heads_ms:.1f} ms")
        print(f"  per-frame aggregator:  {1000 * agg_avg / S:.1f} ms/frame")


if __name__ == "__main__":
    main()
