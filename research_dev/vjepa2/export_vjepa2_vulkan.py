"""
Export vJEPA2 ViT-L to ExecuTorch + Vulkan for OP15 Adreno GPU.
This is the path that should handle conv3d (XNNPACK doesn't, portable
CPU rejects 5D), assuming the Vulkan backend has 3D conv kernels.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoModel

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values_videos).last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/vjepa2/vjepa2_vitl_vulkan_fp32.pte")
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()

    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    print(f"loading {MODEL_ID} ...")
    model = AutoModel.from_pretrained(MODEL_ID).cpu().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")
    wrapper = VJEPA2EncoderWrapper(model)

    H = W = model.config.crop_size
    example = torch.zeros(1, args.frames, 3, H, W, dtype=torch.float32)
    print(f"example input: {tuple(example.shape)} {example.dtype}")

    print("\n[1/3] torch.export ...")
    t0 = time.time()
    with torch.inference_mode():
        ep = torch.export.export(wrapper, (example,), strict=True)
    print(f"  exported in {time.time() - t0:.1f}s")

    print("\n[2/3] to_edge + Vulkan partition ...")
    t0 = time.time()
    et_prog_mgr = to_edge_transform_and_lower(
        ep,
        partitioner=[VulkanPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    print(f"  partitioned in {time.time() - t0:.1f}s")

    print("\n[3/3] to_executorch ...")
    et_prog = et_prog_mgr.to_executorch()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        et_prog.write_to_file(f)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
