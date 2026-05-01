"""
Step 1: Run Qwen2-VL-7B's vision encoder on one image in fp16 on the GPU,
save the resulting embedding as the reference for later comparison.

fp16 (not fp32) is used because that is what the cloud serving stack
(vLLM / HF transformers default) will actually run, so it is the honest
target for the on-device fp16/INT8 deltas.

Output: research_dev/edge_encoder/baseline/{ref_fp16.pt, ref_meta.json}
"""
import argparse
import io
import json
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
# A stable, well-known COCO image (two cats on a couch).
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE_URL)
    ap.add_argument("--out", default="research_dev/edge_encoder/baseline")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} in fp16 on cuda (this downloads ~14 GB on first run)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    img = load_image(args.image)
    print(f"image: {img.size} ({img.mode})")

    inputs = processor.image_processor(images=[img], return_tensors="pt").to("cuda")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    print(f"pixel_values: {tuple(pixel_values.shape)} {pixel_values.dtype}")
    print(f"image_grid_thw: {image_grid_thw.tolist()}")

    with torch.inference_mode():
        embeds = model.model.visual(pixel_values, grid_thw=image_grid_thw)
    print(f"embeds:        {tuple(embeds.shape)} {embeds.dtype}")

    torch.save(
        {
            "embeds": embeds.cpu(),
            "pixel_values": pixel_values.cpu(),
            "image_grid_thw": image_grid_thw.cpu(),
        },
        out / "ref_fp16.pt",
    )
    meta = {
        "model_id": MODEL_ID,
        "image_source": args.image,
        "image_size": list(img.size),
        "pixel_values_shape": list(pixel_values.shape),
        "image_grid_thw": image_grid_thw.tolist(),
        "embed_shape": list(embeds.shape),
        "embed_dtype": str(embeds.dtype),
        "embed_norm": float(embeds.float().norm().item()),
    }
    (out / "ref_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}/ref_fp16.pt")
    print(f"saved {out}/ref_meta.json")


if __name__ == "__main__":
    main()
