"""
Step 2: torch.export Qwen2-VL's ViT, lower to ExecuTorch with NO backend
(portable runtime), run, and compare against the Step 1 fp16 baseline.

Static shapes only — uses the same (pixel_values, image_grid_thw) saved by
extract_vit_baseline.py. Goal is to isolate graph-capture / op-coverage
issues from QNN-specific issues that come in Step 3.
"""
import argparse
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    VisionAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def _exportable_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb=None,
    position_embeddings=None,
    **kwargs,
) -> torch.Tensor:
    """Single-image attention. Skips cu_seqlens.tolist() (data-dependent)."""
    seq_length = hidden_states.shape[0]
    qkv = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
    )
    query_states, key_states, value_states = qkv.unbind(0)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    attn_output, _ = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0,
        is_causal=False,
        **kwargs,
    )
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


def patch_vision_attention(visual: torch.nn.Module) -> int:
    """Replace forward on every VisionAttention submodule. Returns count."""
    n = 0
    for m in visual.modules():
        if isinstance(m, VisionAttention):
            m.forward = _exportable_vision_attention_forward.__get__(m, VisionAttention)
            n += 1
    return n


class VisualOnly(torch.nn.Module):
    """Static-shape wrapper. Precomputes everything derived from grid_thw
    (rotary cos/sin, cu_seqlens) so the exported graph only depends on
    pixel_values. Mirrors Qwen2VisionTransformerPretrainedModel.forward."""

    def __init__(self, visual: torch.nn.Module, grid_thw: torch.Tensor):
        super().__init__()
        self.patch_embed = visual.patch_embed
        self.blocks = visual.blocks
        self.merger = visual.merger
        with torch.no_grad():
            rotary_pos_emb = visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            pos_cos = emb.cos()
            pos_sin = emb.sin()
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        self.register_buffer("pos_cos", pos_cos, persistent=False)
        self.register_buffer("pos_sin", pos_sin, persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(pixel_values)
        position_embeddings = (self.pos_cos, self.pos_sin)
        for blk in self.blocks:
            h = blk(
                h,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return self.merger(h)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="research_dev/edge_encoder/baseline/ref_fp16.pt")
    ap.add_argument("--out", default="research_dev/edge_encoder/exported")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = torch.load(args.baseline, map_location="cpu", weights_only=True)
    pixel_values = ref["pixel_values"].float()      # (1564, 1176) fp32
    image_grid_thw = ref["image_grid_thw"]          # int64 (1, 3)
    ref_embeds = ref["embeds"].float()              # (391, 3584)
    print(f"pixel_values:   {tuple(pixel_values.shape)} {pixel_values.dtype}")
    print(f"image_grid_thw: {image_grid_thw.tolist()}")
    print(f"ref embeds:     {tuple(ref_embeds.shape)} (fp16-cuda baseline)")

    print(f"\nLoading {MODEL_ID} (cached weights) and extracting visual...")
    full = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    visual = full.model.visual.float().cpu().eval()
    del full

    n_patched = patch_vision_attention(visual)
    print(f"  patched {n_patched} VisionAttention modules (single-image static path)")

    wrapper = VisualOnly(visual, image_grid_thw).eval()

    print("\n[stage 1] eager fp32 cpu forward (parity check)...")
    with torch.inference_mode():
        eager_out = wrapper(pixel_values)
    print(f"  eager out: {tuple(eager_out.shape)} {eager_out.dtype}")
    print(f"  cosine(eager_fp32_cpu, ref_fp16_cuda): {cos_sim(eager_out, ref_embeds):.6f}")

    print("\n[stage 2] torch.export.export ...")
    ep = torch.export.export(wrapper, (pixel_values,))
    print(f"  exported program: {len(list(ep.graph.nodes))} nodes")

    print("\n[stage 3] lower to ExecuTorch (no backend partitioner)...")
    edge = to_edge_transform_and_lower(
        ep,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    et = edge.to_executorch()
    pte_path = out / "qwen2vl_vit_portable.pte"
    pte_path.write_bytes(et.buffer)
    size_mb = pte_path.stat().st_size / 1e6
    print(f"  wrote {pte_path} ({size_mb:.1f} MB)")

    print("\n[stage 4] run .pte via portable runtime ...")
    rt = Runtime.get()
    program = rt.load_program(str(pte_path))
    method = program.load_method("forward")
    pte_out = method.execute([pixel_values])[0]
    print(f"  pte out: {tuple(pte_out.shape)} {pte_out.dtype}")

    print("\n=== similarity summary ===")
    print(f"  eager_fp32_cpu  vs ref_fp16_cuda: {cos_sim(eager_out, ref_embeds):.6f}")
    print(f"  pte_portable    vs ref_fp16_cuda: {cos_sim(pte_out,  ref_embeds):.6f}")
    print(f"  pte_portable    vs eager_fp32_cpu:{cos_sim(pte_out,  eager_out):.6f}")


if __name__ == "__main__":
    main()
