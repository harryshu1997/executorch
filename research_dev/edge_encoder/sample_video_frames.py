"""
Sample frames from a video at target fps, preprocess for Qwen2-VL ViT
(static-shape path: resize to 644x476 -> 1564 patches), and write
per-frame .raw + input_list.txt ready to push to the phone and run via
qnn_executor_runner.
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from transformers import AutoProcessor

MODEL_CONFIG = {
    "qwen2vl": {
        "id": "Qwen/Qwen2-VL-7B-Instruct",
        # 46 * 14 x 34 * 14 -> 1564 patches
        "resize": (644, 476),
        "expected_shape": (1564, 1176),
    },
    "llava": {
        "id": "llava-hf/llava-1.5-7b-hf",
        # CLIP processor resizes internally; the processor picks 336x336.
        "resize": None,
        "expected_shape": (1, 3, 336, 336),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video (mp4/mkv/...).")
    ap.add_argument("--fps", type=float, default=1.0, help="Target sampling rate.")
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--model", default="qwen2vl", choices=list(MODEL_CONFIG))
    ap.add_argument("--out", default=None,
                    help="Default: research_dev/edge_encoder/<model>/video_frames.")
    args = ap.parse_args()
    cfg = MODEL_CONFIG[args.model]
    if args.out is None:
        args.out = f"research_dev/edge_encoder/{args.model}/video_frames"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.video} ...")
    meta = iio.immeta(args.video, plugin="pyav")
    src_fps = float(meta.get("fps", 0) or 0)
    if src_fps <= 0:
        raise RuntimeError("Could not determine source fps.")
    step = max(1, int(round(src_fps / args.fps)))

    sampled = []
    for i, frame in enumerate(iio.imiter(args.video, plugin="pyav")):
        if i % step == 0:
            sampled.append(frame)  # HxWx3 uint8
            if len(sampled) >= args.max_frames:
                break
    print(f"  source fps {src_fps:.2f}, step {step} -> {len(sampled)} frames @ ~{args.fps} fps")

    processor = AutoProcessor.from_pretrained(cfg["id"])

    rel_prefix = out.name
    lines = []
    total_bytes = 0
    expected = cfg["expected_shape"]
    for i, t in enumerate(sampled):
        img = Image.fromarray(np.asarray(t))
        if cfg["resize"] is not None:
            img = img.resize(cfg["resize"], Image.LANCZOS)
        pv = processor.image_processor(images=[img], return_tensors="pt")["pixel_values"]
        if tuple(pv.shape) != expected:
            raise RuntimeError(f"frame {i}: expected {expected}, got {tuple(pv.shape)}")
        fname = f"frame_{i:04d}.raw"
        pv.float().numpy().tofile(out / fname)
        total_bytes += (out / fname).stat().st_size
        lines.append(f"{rel_prefix}/{fname}")

    (out / "input_list.txt").write_text("\n".join(lines) + "\n")
    print(f"  wrote {len(lines)} .raw files ({total_bytes/1e9:.2f} GB) + input_list.txt to {out}")


if __name__ == "__main__":
    main()
