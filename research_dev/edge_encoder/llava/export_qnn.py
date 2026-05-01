"""
LLaVA-1.5 Step 2/3: export CLIP ViT-L/14-336 + multi_modal_projector to
QNN HTP. Supports fp16 (default) and pt2e 8a8w.

Graph: pixel_values (1,3,336,336) -> ViT encoder (first 23 of 24 layers)
       -> drop CLS token -> multi_modal_projector (2-layer MLP) -> (1,576,4096)
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.utils import make_quantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


class VisualOnly(nn.Module):
    """pixel_values -> 23 CLIP layers -> drop CLS -> [mm_projector] -> out.

    If include_projector=False, output is the ViT features (1, 576, 1024),
    and the mm_projector is run externally (on host, in fp16). This keeps
    the quantized graph small and avoids the 4096-dim output range blowing
    INT8 scales.
    """

    def __init__(self, full_model: nn.Module, include_projector: bool = True):
        super().__init__()
        vt = full_model.model.vision_tower.vision_model
        self.embeddings = vt.embeddings
        self.pre_layrnorm = vt.pre_layrnorm
        self.layers = nn.ModuleList(vt.encoder.layers[:-1])
        self.include_projector = include_projector
        if include_projector:
            self.mm_projector = full_model.model.multi_modal_projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(pixel_values)
        h = self.pre_layrnorm(h)
        for layer in self.layers:
            h = layer(h, attention_mask=None)
        h = h[:, 1:]  # drop CLS
        if self.include_projector:
            h = self.mm_projector(h)
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="research_dev/edge_encoder/llava/baseline/ref_fp16.pt")
    ap.add_argument("--out", default="research_dev/edge_encoder/llava/qnn_fp16_op12")
    ap.add_argument("--soc", default="SM8650")
    ap.add_argument("--quant_dtype", default=None, choices=[None, "use_8a8w", "use_16a8w"])
    ap.add_argument("--n_calib", type=int, default=20)
    ap.add_argument(
        "--backbone_only",
        action="store_true",
        help="Export only the CLIP ViT backbone (no mm_projector). Output is "
             "(1,576,1024) ViT features; run mm_projector on host.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = torch.load(args.baseline, map_location="cpu", weights_only=True)
    pixel_values = ref["pixel_values"].float()          # (1, 3, 336, 336)
    ref_tensor_key = "vit_features" if args.backbone_only else "embeds"
    ref_tensor = ref[ref_tensor_key].float()
    print(f"pixel_values: {tuple(pixel_values.shape)} {pixel_values.dtype}")
    print(f"ref {ref_tensor_key}: {tuple(ref_tensor.shape)}")

    print(f"\nLoading {MODEL_ID} (cached) + extracting visual...")
    full = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    wrapper = VisualOnly(full, include_projector=not args.backbone_only).float().cpu().eval()
    del full

    print("\n[sanity] eager fp32 cpu forward vs fp16 cuda baseline...")
    with torch.no_grad():
        eager = wrapper(pixel_values)
    cos = torch.nn.functional.cosine_similarity(
        eager.flatten().unsqueeze(0), ref_tensor.flatten().unsqueeze(0)
    ).item()
    print(f"  eager: {tuple(eager.shape)}  cosine vs ref {ref_tensor_key}: {cos:.6f}")

    # Save raw input for qnn_executor_runner.
    raw_path = out / "pixel_values.raw"
    pixel_values.numpy().tofile(raw_path)
    print(f"  wrote {raw_path} ({raw_path.stat().st_size/1e6:.1f} MB)")

    if args.quant_dtype:
        print(f"\nQuantizing with pt2e {args.quant_dtype}, {args.n_calib} calib images...")
        from transformers.models.clip.image_processing_clip import CLIPImageProcessor
        from PIL import Image
        import requests, io

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        coco_ids = [39769, 397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985,
                    458054, 331352, 296649, 386912, 502136, 491497, 184791, 348881,
                    289393, 522713, 181666]
        cache = Path("research_dev/edge_encoder/coco_cache")
        cache.mkdir(parents=True, exist_ok=True)
        calib = []
        for cid in coco_ids[: args.n_calib]:
            local = cache / f"{cid:012d}.jpg"
            if not local.exists():
                url = f"http://images.cocodataset.org/val2017/{cid:012d}.jpg"
                local.write_bytes(requests.get(url, timeout=30).content)
            img = Image.open(local).convert("RGB")
            pv = processor.image_processor(images=[img], return_tensors="pt")["pixel_values"]
            assert tuple(pv.shape) == (1, 3, 336, 336)
            calib.append((pv.float(),))
        print(f"  built {len(calib)} calibration tensors")

        captured = torch.export.export(wrapper, (pixel_values,), strict=False).module()
        qd = getattr(QuantDtype, args.quant_dtype)
        quantizer = make_quantizer(
            quant_dtype=qd, soc_model=args.soc, per_channel_linear=True
        )
        annotated = prepare_pt2e(captured, quantizer)
        print("  calibrating...")
        with torch.no_grad():
            for i, (pv,) in enumerate(calib):
                annotated(pv)
                print(f"    [{i+1}/{len(calib)}]")
        quantized = convert_pt2e(annotated)
        to_lower = quantized
        use_fp16_htp = False
    else:
        to_lower = wrapper
        use_fp16_htp = True

    print("\nBuilding QNN compile spec...")
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=generate_htp_compiler_spec(use_fp16=use_fp16_htp),
    )
    print("Lowering to QNN...")
    edge = to_edge_transform_and_lower_to_qnn(to_lower, (pixel_values,), compile_spec)
    et = edge.to_executorch()
    suffix = args.quant_dtype if args.quant_dtype else "fp16"
    pte_path = out / f"llava_clip_qnn_{suffix}.pte"
    pte_path.write_bytes(et.buffer)
    print(f"\n=== success ===")
    print(f"  pte: {pte_path} ({pte_path.stat().st_size/1e6:.1f} MB)")
    print(f"  raw: {raw_path}")


if __name__ == "__main__":
    main()
