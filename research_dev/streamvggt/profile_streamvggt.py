"""
Profile StreamVGGT's per-frame streaming cost vs VGGT's full-sequence forward.

What this measures, on A6000 (server tier), for S ∈ {1, 4, 8, 16}:

  - streaming `inference()` per-frame latency (should be ~constant in S).
  - full `forward()` total latency (reference; should grow O(S²) like VGGT).
  - head-output bytes per frame (what a phone would upload if it ran the
    whole model locally).
  - aggregator-token bytes per frame (what a phone would upload if we split
    at the aggregator/heads boundary).

This populates the `(StreamVGGT, A6000, {})` cost-table row
(MULTI_MODEL_PLAN §4) and also gives us the upper-tier reference for the
per-device rows once the aggregator is exported to QNN.
"""
import sys
import time
from pathlib import Path

import torch

STREAMVGGT_ROOT = Path("/home/myid/zs89458/Documents/StreamVGGT")
VGGT_ROOT = Path("/home/myid/zs89458/Documents/vggt")
for p in (STREAMVGGT_ROOT / "src", VGGT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402


def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def dict_bytes(d: dict) -> int:
    return sum(tensor_bytes(v) for v in d.values() if isinstance(v, torch.Tensor))


def kv_cache_bytes(pkv) -> int:
    """Total bytes held in the aggregator's past_key_values list."""
    total = 0
    for entry in pkv:
        if entry is None:
            continue
        k, v = entry
        total += tensor_bytes(k) + tensor_bytes(v)
    return total


def time_fn(fn, warm: int = 1, runs: int = 3) -> tuple[float, float, float]:
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
    print(f"loading StreamVGGT on {device}...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").to(device).eval()

    img_dir = VGGT_ROOT / "examples" / "kitchen" / "images"
    img_paths = sorted(img_dir.glob("*.png"))[:16]
    print(f"preprocessing {len(img_paths)} kitchen frames...")
    images, _ = load_and_preprocess_images_square(
        [str(p) for p in img_paths], target_size=518
    )
    images = images.to(device)  # (16, 3, 518, 518)

    # One-time: size of per-frame head output and aggregator tokens.
    with torch.inference_mode():
        out = model.inference([{"img": images[0:1]}])
    per_frame_head_bytes = dict_bytes(out.ress[0])
    print(f"\nper-frame head output: {per_frame_head_bytes/1e6:.2f} MB")
    print("  breakdown:")
    for k, v in out.ress[0].items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {tuple(v.shape)} → {tensor_bytes(v)/1e6:.3f} MB")

    # Aggregator-only token size for one frame (what we'd send to server if
    # we split at aggregator/heads). Need to call the aggregator directly.
    with torch.inference_mode():
        agg_out = model.aggregator(images[0:1].unsqueeze(0))  # (1, 1, 3, H, W)
    # aggregator returns (tokens_list, patch_start_idx) for non-cached path.
    tokens_list = agg_out[0] if isinstance(agg_out, tuple) else agg_out
    if isinstance(tokens_list, list):
        agg_bytes = sum(tensor_bytes(t) for t in tokens_list)
        shapes = [tuple(t.shape) for t in tokens_list[:2]]
    else:
        agg_bytes = tensor_bytes(tokens_list)
        shapes = [tuple(tokens_list.shape)]
    print(f"\nper-frame aggregator tokens: {agg_bytes/1e6:.2f} MB  (shapes: {shapes}, ...)")

    # Timing sweep.
    for S in (1, 4, 8, 16):
        frames = [{"img": images[i : i + 1]} for i in range(S)]
        views = [{"img": images[i : i + 1]} for i in range(S)]

        def streaming():
            with torch.inference_mode():
                model.inference(frames)

        def full_forward():
            with torch.inference_mode():
                model(views)

        str_min, str_avg, str_max = time_fn(streaming)
        full_min, full_avg, full_max = time_fn(full_forward)

        print(f"\n--- S = {S} ---")
        print(f"  streaming inference(): {1000*str_avg:.1f} ms total "
              f"({1000*str_avg/S:.1f} ms/frame)  "
              f"[min {1000*str_min:.1f}, max {1000*str_max:.1f}]")
        print(f"  full forward()       : {1000*full_avg:.1f} ms total "
              f"({1000*full_avg/S:.1f} ms/frame)  "
              f"[min {1000*full_min:.1f}, max {1000*full_max:.1f}]")

    # ---- KV-cache growth (stateful-placement cost term) -------------------
    # Stream frames one at a time, mirroring StreamVGGT.inference(), and track
    # (a) per-frame marginal latency and (b) total cache bytes after each frame.
    # Cache bytes = migration cost if the aggregator moves device mid-session.
    print("\n--- KV cache growth (per-frame streaming) ---")
    S_max = 16
    pkv = [None] * model.aggregator.depth
    cumulative_lat_ms = 0.0
    last_head_outputs = None
    print("  frame  marginal_ms  cumul_ms  cache_MB  head_out_MB")
    for i in range(S_max):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            img = images[i : i + 1].unsqueeze(0)  # (1, 1, 3, H, W)
            agg_out = model.aggregator(
                img, past_key_values=pkv, use_cache=True, past_frame_idx=i
            )
            # aggregator returns (tokens_list, patch_start_idx) — pkv is mutated in-place.
            if isinstance(agg_out, tuple) and len(agg_out) == 3:
                tokens_list, patch_start_idx, pkv = agg_out
            else:
                tokens_list, patch_start_idx = agg_out
            # Run heads to get deliverable outputs (bytes-on-wire for phone-only placement).
            depth, depth_conf = model.depth_head(
                tokens_list, images=img, patch_start_idx=patch_start_idx
            )
            pts3d, pts3d_conf = model.point_head(
                tokens_list, images=img, patch_start_idx=patch_start_idx
            )
            pose_enc_list = model.camera_head(tokens_list)
            last_head_outputs = {
                "depth": depth[:, 0],
                "depth_conf": depth_conf[:, 0],
                "world_points": pts3d[:, 0],
                "world_points_conf": pts3d_conf[:, 0],
                "camera_pose": pose_enc_list[-1][:, 0],
            }
        torch.cuda.synchronize()
        marginal_ms = 1000 * (time.time() - t0)
        cumulative_lat_ms += marginal_ms
        c_bytes = kv_cache_bytes(pkv)
        head_bytes = dict_bytes(last_head_outputs)
        print(f"  {i:>5}  {marginal_ms:>10.1f}  {cumulative_lat_ms:>8.1f}  "
              f"{c_bytes/1e6:>8.2f}  {head_bytes/1e6:>10.2f}")

    per_frame_cache_growth = (
        kv_cache_bytes(pkv) / S_max  # linear fit assumed; good enough for sanity.
    )
    print(f"\n  avg cache bytes per frame (linear fit): {per_frame_cache_growth/1e6:.2f} MB")
    print(f"  total cache after {S_max} frames:         {kv_cache_bytes(pkv)/1e6:.2f} MB")

    print("\n--- cost-table row summary (StreamVGGT, A6000, fp32, solo) ---")
    print(f"  per-frame head output:      {per_frame_head_bytes/1e6:.2f} MB")
    print(f"  per-frame aggregator bytes: {agg_bytes/1e6:.2f} MB (24 layers) / "
          f"{agg_bytes/1e6/24:.2f} MB (final layer only)")
    print(f"  per-frame KV-cache growth:  {per_frame_cache_growth/1e6:.2f} MB "
          f"→ migration cost if aggregator changes device")
    print("  (Compare to VGGT: 67 MB/frame aggregator, O(S^2) growth.)")


if __name__ == "__main__":
    main()
