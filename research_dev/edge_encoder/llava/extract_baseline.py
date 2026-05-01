"""
LLaVA-1.5-7B Step 1: run CLIP ViT-L/14-336 + multi_modal_projector on one
image in fp16 on GPU, save reference embedding for later cosine comparison.

Output: research_dev/edge_encoder/llava/baseline/{ref_fp16.pt, ref_meta.json}
"""
import argparse
import io
import json
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # cats


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE_URL)
    ap.add_argument("--out", default="research_dev/edge_encoder/llava/baseline")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} fp16 on cuda (~14 GB first-run download)...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    img = load_image(args.image)
    print(f"image: {img.size} ({img.mode})")

    pv = processor.image_processor(images=[img], return_tensors="pt")["pixel_values"].to("cuda")
    print(f"pixel_values: {tuple(pv.shape)} {pv.dtype}")

    # LLaVA-1.5 pipeline: vision_tower -> select penultimate-layer hidden -> drop CLS -> mm_projector
    # See transformers' LlavaForConditionalGeneration.get_image_features.
    with torch.inference_mode():
        vt_out = model.model.vision_tower(pv, output_hidden_states=True)
        feats = vt_out.hidden_states[-2][:, 1:]  # (1, 576, 1024)
        embeds = model.model.multi_modal_projector(feats)  # (1, 576, 4096)

    print(f"vit features: {tuple(feats.shape)} {feats.dtype}")
    print(f"embeds:       {tuple(embeds.shape)} {embeds.dtype}")

    torch.save(
        {
            "pixel_values": pv.cpu(),
            "vit_features": feats.cpu(),
            "embeds": embeds.cpu(),
        },
        out / "ref_fp16.pt",
    )
    meta = {
        "model_id": MODEL_ID,
        "image_source": args.image,
        "image_size": list(img.size),
        "pixel_values_shape": list(pv.shape),
        "vit_features_shape": list(feats.shape),
        "embeds_shape": list(embeds.shape),
        "embed_dtype": str(embeds.dtype),
        "embed_norm": float(embeds.float().norm().item()),
    }
    (out / "ref_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}/ref_fp16.pt")
    print(f"saved {out}/ref_meta.json")


if __name__ == "__main__":
    main()
