"""
Step 3: Lower Qwen2-VL ViT to QNN HTP (fp16) and produce a .pte.

Reuses the patches from export_vit_pte.py to make the graph exportable, then
swaps the no-backend lowering for QnnPartitioner via the official helper
to_edge_transform_and_lower_to_qnn.

Compile-only by default. To run on the HTP x86 emulator, see the printed
command at the end (uses build-x86/.../qnn_executor_runner).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration

# Reuse the wrapper + monkey-patch from Step 2.
sys.path.insert(0, str(Path(__file__).parent))
from export_vit_pte import VisualOnly, patch_vision_attention  # noqa: E402

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset  # noqa: E402
from executorch.backends.qualcomm.utils.utils import (  # noqa: E402
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="research_dev/edge_encoder/baseline/ref_fp16.pt")
    ap.add_argument("--out", default="research_dev/edge_encoder/qnn_fp16")
    ap.add_argument("--soc", default="SM8550", help="SM8550 = SD8 Gen2 / OP11; SM8650 = SD8 Gen3 / OP12")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = torch.load(args.baseline, map_location="cpu", weights_only=True)
    pixel_values = ref["pixel_values"].float()       # (1564, 1176) fp32
    image_grid_thw = ref["image_grid_thw"]           # int64 (1, 3)
    print(f"pixel_values:   {tuple(pixel_values.shape)} {pixel_values.dtype}")
    print(f"image_grid_thw: {image_grid_thw.tolist()}")

    print(f"\nLoading {MODEL_ID} (cached) and extracting visual...")
    full = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    visual = full.model.visual.float().cpu().eval()
    del full
    n = patch_vision_attention(visual)
    print(f"  patched {n} VisionAttention modules")
    wrapper = VisualOnly(visual, image_grid_thw).eval()

    # Save a raw input file for the qnn_executor_runner.
    raw_path = out / "pixel_values.raw"
    pixel_values.numpy().tofile(raw_path)
    print(f"  wrote {raw_path} ({raw_path.stat().st_size/1e6:.1f} MB)")

    print("\nBuilding QNN compile spec (HTP, fp16)...")
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=backend_options,
    )

    print("\nLowering to QNN ...")
    edge = to_edge_transform_and_lower_to_qnn(
        wrapper,
        (pixel_values,),
        compile_spec,
    )

    print("Serializing to ExecuTorch ...")
    et = edge.to_executorch()
    pte_path = out / "qwen2vl_vit_qnn_fp16.pte"
    pte_path.write_bytes(et.buffer)
    print(f"\n=== success ===")
    print(f"  pte:  {pte_path} ({pte_path.stat().st_size/1e6:.1f} MB)")
    print(f"  raw:  {raw_path}")
    print()
    print("To run on the HTP x86 emulator:")
    print()
    print(f"  echo {raw_path.resolve()} > {out}/input_list.txt")
    print(f"  mkdir -p {out}/emulator_outputs")
    print(
        f"  LD_LIBRARY_PATH=$PWD/build-x86/lib:$QNN_SDK_ROOT/lib/x86_64-linux-clang \\\n"
        f"    ./build-x86/examples/qualcomm/executor_runner/qnn_executor_runner \\\n"
        f"      --model_path {pte_path} \\\n"
        f"      --input_list_path {out}/input_list.txt \\\n"
        f"      --output_folder_path {out}/emulator_outputs"
    )


if __name__ == "__main__":
    main()
