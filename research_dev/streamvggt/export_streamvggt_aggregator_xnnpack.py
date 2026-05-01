"""
Lower StreamVGGT's aggregator (S=1, no KV cache) to ExecuTorch + XNNPACK and
emit a .pte that runs on OP12's ARM CPU. Goal: *make it run at all* so we
have an on-device latency number for the cost table. Energy/quant comes later.

Output: a .pte file. If XNNPACK lowering fails on some op, we fall back to the
portable kernels only (still CPU, just without the XNNPACK accelerated path).
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

STREAMVGGT_ROOT = Path("/home/myid/zs89458/Documents/StreamVGGT")
sys.path.insert(0, str(STREAMVGGT_ROOT / "src"))

from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from streamvggt.layers import rope as _rope  # noqa: E402


def _patch_rope_for_export(patch_grid: int = 40) -> None:
    """Rope's forward does `int(positions.max())+1` which breaks torch.export.
    For fixed image size, patch grid is static; hardcode a max-position bound.
    """
    orig_forward = _rope.RotaryPositionEmbedding2D.forward

    def exportable_forward(self, tokens: torch.Tensor, positions: torch.Tensor):
        feature_dim = tokens.size(-1) // 2
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, patch_grid, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    _rope.RotaryPositionEmbedding2D.forward = exportable_forward
    print(f"  patched rope.forward: max_position hardcoded to {patch_grid}")


class AggregatorWrapper(nn.Module):
    """Tensor-in, tensor-out wrapper so the graph can export cleanly."""

    def __init__(self, model: StreamVGGT):
        super().__init__()
        self.agg = model.aggregator

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B=1, S=1, 3, 518, 518). Returns last-layer tokens.
        tokens_list, _patch_start_idx = self.agg(images)
        return tokens_list[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/streamvggt/streamvggt_agg_xnnpack.pte")
    ap.add_argument("--fp16", action="store_true", help="Cast model + inputs to fp16.")
    ap.add_argument("--no_xnnpack", action="store_true",
                    help="Skip XNNPACK partitioning (portable kernels only).")
    args = ap.parse_args()

    _patch_rope_for_export(patch_grid=40)

    print("loading StreamVGGT on cpu ...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").cpu().eval()
    dtype = torch.float16 if args.fp16 else torch.float32
    model = model.to(dtype)
    wrapper = AggregatorWrapper(model)

    example = torch.zeros(1, 1, 3, 518, 518, dtype=dtype)
    print(f"example input: {tuple(example.shape)} {example.dtype}")

    print("\n[1/3] torch.export...")
    t0 = time.time()
    with torch.inference_mode():
        ep = torch.export.export(wrapper, (example,))
    print(f"  exported in {time.time()-t0:.1f}s")

    print("\n[2/3] to_edge + XNNPACK partition + to_executorch...")
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    t0 = time.time()
    partitioners = []
    if not args.no_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        partitioners.append(XnnpackPartitioner())
    et_prog_mgr = to_edge_transform_and_lower(
        ep,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    et_prog = et_prog_mgr.to_executorch()
    print(f"  lowered in {time.time()-t0:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        et_prog.write_to_file(f)
    size_mb = out.stat().st_size / 1e6
    print(f"\n[3/3] wrote {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
