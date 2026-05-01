"""
VGGT smoke test — load the 1B model from HF, run forward on a few images
from vggt's own example set, print output shapes and timing. No COLMAP
export, no trimesh, no heads disabled. If this works, Phase 1 of
VGGT_PLAN.md is unblocked.
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  torch: {torch.__version__}")

    # Pick the kitchen scene; 3 images is enough for a smoke test.
    img_dir = VGGT_ROOT / "examples" / "kitchen" / "images"
    img_paths = sorted(img_dir.glob("*"))[:3]
    print(f"using {len(img_paths)} images from {img_dir}")
    for p in img_paths:
        print(f"  {p.name}")

    # VGGT expects a (S, 3, H, W) tensor, values in [0, 1].
    print("\npreprocessing...")
    images, _ = load_and_preprocess_images_square([str(p) for p in img_paths], target_size=518)
    images = images.to(device)
    print(f"  images: {tuple(images.shape)} {images.dtype}")

    print("\nloading model (HF: facebook/VGGT-1B)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")

    # Warm run
    print("\nwarm-up forward...")
    with torch.inference_mode():
        model(images)
    torch.cuda.synchronize()

    # Timed run
    print("\ntimed forward x 3...")
    ts = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model(images)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    print(f"  avg latency: {1000 * sum(ts) / len(ts):.1f} ms  (min {1000 * min(ts):.1f}, max {1000 * max(ts):.1f})")

    print("\noutput shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            print(f"  {k}: list[{len(v)}] of {tuple(v[0].shape)}")
        else:
            print(f"  {k}: {type(v).__name__}")


if __name__ == "__main__":
    main()
