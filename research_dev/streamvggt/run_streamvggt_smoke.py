"""
StreamVGGT smoke test — load the model from HF, run its streaming
`inference()` on a few kitchen-scene frames, print output shapes and
per-frame timing. If this works, StreamVGGT is a drop-in candidate to
replace VGGT in MULTI_MODEL_PLAN §2.

Parallels `research_dev/vggt_split/run_vggt_smoke.py`.
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  torch: {torch.__version__}")

    img_dir = VGGT_ROOT / "examples" / "kitchen" / "images"
    img_paths = sorted(img_dir.glob("*.png"))[:6]
    print(f"using {len(img_paths)} images from {img_dir}")
    for p in img_paths:
        print(f"  {p.name}")

    print("\npreprocessing...")
    images, _ = load_and_preprocess_images_square(
        [str(p) for p in img_paths], target_size=518
    )
    images = images.to(device)  # (S, 3, 518, 518)
    # inference() does frame["img"].unsqueeze(0) expecting (B, S, C, H, W),
    # so each frame["img"] must already include the batch dim: (1, 3, H, W).
    frames = [{"img": images[i : i + 1]} for i in range(images.shape[0])]
    print(f"  images: {tuple(images.shape)} {images.dtype}")

    print("\nloading model (HF: lch01/StreamVGGT)...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")

    print("\nwarm-up streaming inference (1 frame)...")
    with torch.inference_mode():
        model.inference([frames[0]])
    torch.cuda.synchronize()

    print("\ntimed streaming inference over S=6...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model.inference(frames)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  total: {1000*dt:.1f} ms  ({1000*dt/len(frames):.1f} ms/frame avg)")

    print("\noutput structure:")
    print(f"  n_results: {len(out.ress)}")
    r0 = out.ress[0]
    for k, v in r0.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {tuple(v.shape)} {v.dtype} ({v.numel()*v.element_size()} B)")
        else:
            print(f"    {k}: {type(v).__name__}")


if __name__ == "__main__":
    main()
