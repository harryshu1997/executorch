"""
Step 4: pt2e INT8 (8a8w) quantize Qwen2-VL ViT and lower to QNN HTP.

Calibration: ~20 COCO val images, each resized to (644x476) to produce
exactly the same 1564 patches / (1564, 1176) pixel_values shape as our
static graph.

Output: research_dev/edge_encoder/qnn_int8_op12/qwen2vl_vit_qnn_int8.pte
"""
import argparse
import io
import sys
from pathlib import Path

import requests
import torch
from PIL import Image
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from export_vit_pte import VisualOnly, patch_vision_attention  # noqa: E402

from executorch.backends.qualcomm.quantizer.quantizer import (  # noqa: E402
    ModuleQConfig,
    QuantDtype,
    get_submodule_type_predicate,
)
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset  # noqa: E402
from executorch.backends.qualcomm.utils.utils import (  # noqa: E402
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.utils import make_quantizer  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


class _LinearPatchEmbed(torch.nn.Module):
    """Mathematically equivalent Linear replacement for Qwen2VL PatchEmbed.

    The original uses Conv3d with kernel == input-tile size; for a flattened
    (N, C*T*H*W) input that's just a matmul. Avoids the 5-D tensor that
    breaks QNN's quantized graph finalize.
    """

    def __init__(self, orig: torch.nn.Module):
        super().__init__()
        in_dim = orig.in_channels * orig.temporal_patch_size * orig.patch_size ** 2
        self.proj = torch.nn.Linear(in_dim, orig.embed_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(orig.proj.weight.reshape(orig.embed_dim, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def replace_patch_embed_with_linear(visual: torch.nn.Module) -> bool:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed

    if isinstance(visual.patch_embed, PatchEmbed):
        visual.patch_embed = _LinearPatchEmbed(visual.patch_embed)
        return True
    return False
TARGET_WH = (644, 476)  # (w, h) -> 46 x 34 patches = 1564

# 20 COCO val2017 image IDs (stable, public).
COCO_IDS = [
    39769, 397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985, 458054,
    331352, 296649, 386912, 502136, 491497, 184791, 348881, 289393, 522713, 181666,
]
COCO_URL = "http://images.cocodataset.org/val2017/{id:012d}.jpg"


def fetch_image(image_id: int, cache: Path) -> Image.Image:
    local = cache / f"{image_id:012d}.jpg"
    if not local.exists():
        r = requests.get(COCO_URL.format(id=image_id), timeout=30)
        r.raise_for_status()
        local.write_bytes(r.content)
    return Image.open(local).convert("RGB").resize(TARGET_WH, Image.LANCZOS)


def build_calibration_set(processor, cache: Path, n: int):
    cache.mkdir(parents=True, exist_ok=True)
    cal = []
    for i, cid in enumerate(COCO_IDS[:n]):
        img = fetch_image(cid, cache)
        pv = processor.image_processor(images=[img], return_tensors="pt")["pixel_values"]
        if tuple(pv.shape) != (1564, 1176):
            raise RuntimeError(f"image {cid}: expected (1564,1176), got {tuple(pv.shape)}")
        cal.append((pv.float(),))
        print(f"  [{i+1}/{n}] coco {cid}: {tuple(pv.shape)}")
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="research_dev/edge_encoder/baseline/ref_fp16.pt")
    ap.add_argument("--out", default="research_dev/edge_encoder/qnn_int8_op12")
    ap.add_argument("--soc", default="SM8650")
    ap.add_argument("--n_calib", type=int, default=20)
    ap.add_argument(
        "--quant_dtype",
        default="use_16a8w",
        choices=["use_8a8w", "use_16a8w", "use_16a4w"],
        help="HTP quantization scheme. 16a8w is typically most robust for transformers.",
    )
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="Keep LayerNorm + QuickGELU + softmax in 16a8w, rest in --quant_dtype. "
             "Only meaningful with --quant_dtype use_8a8w.",
    )
    ap.add_argument(
        "--histogram_observer",
        action="store_true",
        help="Use HistogramObserver (percentile clipping) instead of MinMax for "
             "activation scale selection. Calibration is slower but more robust to outliers.",
    )
    ap.add_argument(
        "--act_symmetric",
        action="store_true",
        help="Use symmetric activation quantization (skip zero-point). "
             "Better for zero-centered distributions (layernorm outputs).",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("research_dev/edge_encoder/coco_cache")

    ref = torch.load(args.baseline, map_location="cpu", weights_only=True)
    pixel_values = ref["pixel_values"].float()
    image_grid_thw = ref["image_grid_thw"]
    print(f"pixel_values (ref): {tuple(pixel_values.shape)} {pixel_values.dtype}")

    print(f"\nLoading {MODEL_ID} (cached) and extracting visual...")
    full = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    visual = full.model.visual.float().cpu().eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    del full
    n = patch_vision_attention(visual)
    print(f"  patched {n} VisionAttention modules")
    if replace_patch_embed_with_linear(visual):
        print("  replaced Conv3d patch_embed with equivalent Linear (for INT8 HTP compat)")
    wrapper = VisualOnly(visual, image_grid_thw).eval()

    # Sanity: confirm the Linear swap is bit-equivalent to Conv3d on this input.
    with torch.no_grad():
        eager_out = wrapper(pixel_values)
    ref_embeds = ref["embeds"].float()
    cos = torch.nn.functional.cosine_similarity(
        eager_out.flatten().unsqueeze(0), ref_embeds.flatten().unsqueeze(0)
    ).item()
    print(f"  eager (Linear patch_embed) vs fp16-cuda ref: cosine={cos:.6f}")

    print(f"\nBuilding calibration set ({args.n_calib} COCO images resized to {TARGET_WH})...")
    calibration = build_calibration_set(processor, cache, args.n_calib)

    # Save the baseline-image raw (for on-device run + comparison).
    raw_path = out / "pixel_values.raw"
    pixel_values.numpy().tofile(raw_path)
    print(f"  wrote {raw_path} ({raw_path.stat().st_size/1e6:.1f} MB)")

    print("\ntorch.export for pt2e...")
    captured = torch.export.export(wrapper, (pixel_values,), strict=False).module()

    qd = getattr(QuantDtype, args.quant_dtype)
    submodule_qconfig_list = None
    if args.hybrid:
        hi = ModuleQConfig(
            quant_dtype=QuantDtype.use_16a8w,
            is_linear_per_channel=True,
        )

        def target_in(targets):
            def pred(node):
                return node.op == "call_function" and node.target in targets
            return pred

        softmax_targets = {
            torch.ops.aten._softmax.default,
            torch.ops.aten.softmax.int,
        }
        submodule_qconfig_list = [
            (get_submodule_type_predicate("LayerNorm"), hi),
            (get_submodule_type_predicate("QuickGELUActivation"), hi),
            (target_in(softmax_targets), hi),
        ]
        print(f"  HYBRID: base={args.quant_dtype}; 16a8w overrides for LayerNorm + QuickGELU + softmax")
    qkwargs = dict(
        quant_dtype=qd,
        soc_model=args.soc,
        per_channel_linear=True,
        submodule_qconfig_list=submodule_qconfig_list,
        act_symmetric=args.act_symmetric,
    )
    if args.histogram_observer:
        from torchao.quantization.pt2e import HistogramObserver
        qkwargs["act_observer"] = HistogramObserver
    print(f"make_quantizer({args.quant_dtype}, soc={args.soc}, per_channel_linear=True, "
          f"act_symmetric={args.act_symmetric}, histogram={args.histogram_observer})...")
    quantizer = make_quantizer(**qkwargs)

    print("prepare_pt2e...")
    annotated = prepare_pt2e(captured, quantizer)

    print(f"\nCalibrating ({len(calibration)} images, CPU fp32, each ~1-2 min)...")
    with torch.no_grad():
        for i, (pv,) in enumerate(calibration):
            annotated(pv)
            print(f"  [{i+1}/{len(calibration)}] done")

    print("\nconvert_pt2e...")
    quantized = convert_pt2e(annotated)

    print("Building QNN compile spec (HTP INT8)...")
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=backend_options,
    )

    print("Lowering quantized model to QNN...")
    edge = to_edge_transform_and_lower_to_qnn(quantized, (pixel_values,), compile_spec)

    print("Serializing to .pte ...")
    et = edge.to_executorch()
    pte_path = out / f"qwen2vl_vit_qnn_{args.quant_dtype}.pte"
    pte_path.write_bytes(et.buffer)
    print(f"\n=== success ===")
    print(f"  pte: {pte_path} ({pte_path.stat().st_size/1e6:.1f} MB)")
    print(f"  raw: {raw_path}")


if __name__ == "__main__":
    main()
