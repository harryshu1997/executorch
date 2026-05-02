"""
Export vJEPA2 ViT-L to Vulkan with conv3d → conv2d rewrite.

The patch_embeddings.proj is `Conv3d(3, 1024, k=(2,16,16), s=(2,16,16))`
with non-overlapping stride. This is mathematically equivalent to
reshaping pairs of frames into the channel dim and running Conv2d:

    input (B, C=3, T=16, H=256, W=256)
        view → (B, T/2=8, 2*C=6, H, W) [pairs of frames as channels]
        merge time into batch: (B*8, 6, H, W)
        conv2d(weight (1024, 6, 16, 16), stride 16) → (B*8, 1024, 16, 16)
        reshape → (B, 1024, 8, 16, 16)

ExecuTorch's Vulkan/XNNPACK/portable backends all skip 5D conv3d. This
rewrite makes the model exportable.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Conv3dToConv2dPatchEmbed(nn.Module):
    """Drop-in replacement for VJEPA2PatchEmbeddings3D.proj that uses
    Conv2d after a frame-pair channel-stack reshape."""

    def __init__(self, conv3d: nn.Conv3d):
        super().__init__()
        out_c, in_c, k_t, k_h, k_w = conv3d.weight.shape
        self.k_t = k_t
        self.k_h = k_h
        self.k_w = k_w
        # Reshape weight (out_c, in_c, k_t, k_h, k_w) → (out_c, in_c*k_t, k_h, k_w)
        # by treating (in_c, k_t) as flattened input channels.
        # PyTorch Conv2d weight layout is (out_c, in_c, k_h, k_w).
        # We want weight2d[oc, c + t * in_c, h, w] = weight3d[oc, c, t, h, w]
        # i.e., permute dims so time slot t is grouped with input channels.
        w3d = conv3d.weight.detach()  # (out_c, in_c, k_t, k_h, k_w)
        # Permute to (out_c, k_t, in_c, k_h, k_w) → flatten (k_t, in_c)
        w2d = w3d.permute(0, 2, 1, 3, 4).reshape(out_c, k_t * in_c, k_h, k_w)
        self.conv2d = nn.Conv2d(
            in_channels=k_t * in_c,
            out_channels=out_c,
            kernel_size=(k_h, k_w),
            stride=(k_h, k_w),  # non-overlapping spatial stride
            bias=conv3d.bias is not None,
        )
        self.conv2d.weight = nn.Parameter(w2d.contiguous())
        if conv3d.bias is not None:
            self.conv2d.bias = nn.Parameter(conv3d.bias.detach().clone())

    @property
    def weight(self):
        # vJEPA2's encoder reads patch_embeddings.proj.weight to infer
        # output channels. Forward the underlying conv2d's weight.
        return self.conv2d.weight

    @property
    def bias(self):
        return self.conv2d.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — same input layout the original conv3d expects.
        B, C, T, H, W = x.shape
        # Reshape: (B, C, T, H, W) → (B, T/k_t, k_t, C, H, W) → (B, T/k_t, k_t*C, H, W)
        # Then merge time-groups into batch for conv2d.
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B, T // self.k_t, self.k_t * C, H, W)
        # Merge batch and time into one batch dim for conv2d.
        x = x.reshape(B * (T // self.k_t), self.k_t * C, H, W)
        x = self.conv2d(x)  # (B*T/k_t, out_c, H/k_h, W/k_w)
        # Reshape back to (B, out_c, T/k_t, H/k_h, W/k_w) to match conv3d output.
        out_c = x.shape[1]
        H_o = H // self.k_h
        W_o = W // self.k_w
        x = x.reshape(B, T // self.k_t, out_c, H_o, W_o)
        x = x.permute(0, 2, 1, 3, 4)  # (B, out_c, T/k_t, H_o, W_o)
        return x.contiguous()


def _replace_patch_embed_conv3d(model) -> None:
    pe = model.encoder.embeddings.patch_embeddings
    new_proj = Conv3dToConv2dPatchEmbed(pe.proj)
    pe.proj = new_proj
    print(f"  replaced patch_embeddings.proj Conv3d → Conv2d-via-reshape")


class VJEPA2EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values_videos).last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/vjepa2/vjepa2_vitl_vulkan_no3d.pte")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--fp16", action="store_true", help="Cast model + inputs to fp16")
    args = ap.parse_args()

    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    print(f"loading {MODEL_ID} ...")
    model = AutoModel.from_pretrained(MODEL_ID).cpu().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")

    # Sanity: verify the conv2d replacement matches conv3d numerically.
    print("\n[0/3] verifying conv3d→conv2d numerical equivalence ...")
    test_in = torch.randn(1, 3, args.frames, 256, 256)
    orig_out = model.encoder.embeddings.patch_embeddings.proj(test_in)
    _replace_patch_embed_conv3d(model)
    new_out = model.encoder.embeddings.patch_embeddings.proj(test_in)
    diff = (orig_out - new_out).abs().max().item()
    print(f"  max abs diff: {diff:.2e}  (should be ~1e-5 from fp accumulation)")
    assert diff < 1e-3, f"conv2d replacement diverged: {diff}"

    if args.fp16:
        model = model.to(torch.float16)
        print("  cast model to fp16")

    wrapper = VJEPA2EncoderWrapper(model)
    H = W = model.config.crop_size
    example_dtype = torch.float16 if args.fp16 else torch.float32
    example = torch.zeros(1, args.frames, 3, H, W, dtype=example_dtype)
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
