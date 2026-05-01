"""
Phase 6 sanity: does LLaVA-1.5-7B's LLM produce a correct answer when fed
image tokens computed on the phone (via our 640 MB QNN fp16 .pte) instead
of tokens computed by the cloud vision tower?

Runs the model in two modes on the same image and prompt, prints both
answers so you can eyeball whether on-device ViT drift breaks semantics.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"


def load_image(path_or_url: str) -> Image.Image:
    import io, requests
    if path_or_url.startswith(("http://", "https://")):
        return Image.open(io.BytesIO(requests.get(path_or_url, timeout=30).content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def generate(model, processor, image, max_new_tokens=60):
    inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda", torch.float16)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    full = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return full.split("ASSISTANT:", 1)[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phone_output",
        default="research_dev/edge_encoder/llava/qnn_fp16_op12/phone_output_fp16.raw",
        help="Phone's .pte output (576*4096 fp32 raw bytes).",
    )
    ap.add_argument(
        "--image",
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="The same image that produced the phone output (cats image by default).",
    )
    ap.add_argument("--max_new_tokens", type=int, default=60)
    args = ap.parse_args()

    print(f"Loading {MODEL_ID} fp16 on cuda...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    img = load_image(args.image)
    print(f"image: {img.size}")

    # --- Cloud baseline ---
    print("\n=== CLOUD-only (vision_tower on A6000) ===")
    cloud_ans = generate(model, processor, img, args.max_new_tokens)
    print(cloud_ans)

    # --- Edge path: inject phone tokens ---
    phone = torch.from_numpy(
        np.fromfile(args.phone_output, dtype=np.float32)
    ).reshape(1, 576, 4096).to("cuda", torch.float16)
    print(f"\nPhone tokens: {tuple(phone.shape)} {phone.dtype} "
          f"(stats: min={phone.min():.2f} max={phone.max():.2f} std={phone.std():.2f})")

    original_get_image_features = model.model.get_image_features

    def patched_get_image_features(*args_, **kwargs_):
        return [phone[0]]  # list with one (576, 4096) tensor

    model.model.get_image_features = patched_get_image_features
    print("\n=== EDGE path (vision_tower run on OP12 HTP fp16) ===")
    edge_ans = generate(model, processor, img, args.max_new_tokens)
    print(edge_ans)

    # Restore
    model.model.get_image_features = original_get_image_features

    print("\n=== COMPARISON ===")
    print(f"cloud: {cloud_ans!r}")
    print(f"edge : {edge_ans!r}")
    if cloud_ans.strip().lower() == edge_ans.strip().lower():
        print(">> identical")
    else:
        # Rough lexical overlap
        c_words = set(cloud_ans.lower().split())
        e_words = set(edge_ans.lower().split())
        overlap = len(c_words & e_words) / max(1, len(c_words | e_words))
        print(f">> different text; jaccard overlap of words = {overlap:.2%}")


if __name__ == "__main__":
    main()
