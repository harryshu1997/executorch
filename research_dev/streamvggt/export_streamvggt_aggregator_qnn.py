"""
Lower StreamVGGT's aggregator (S=1, no KV cache) to ExecuTorch + QNN HTP for
OP15 (SM8850, Hexagon v81, fp16). Goal: produce a .pte that runs on the NPU,
so we can replace the placeholder phone-tier latency in cost_table.py.

Mirrors research_dev/streamvggt/export_streamvggt_aggregator_xnnpack.py but
swaps the partitioner: XNNPACK → QNN. Reuses the rope monkey-patch.

Run with QNN env vars set, e.g.:
  PATH=$HOME/miniforge3/envs/research/bin:$PATH \
  QNN_SDK_ROOT=$HOME/qairt/2.45.0.260326 \
  LD_LIBRARY_PATH=$HOME/qairt/2.45.0.260326/lib/x86_64-linux-clang \
  PYTHONPATH=$HOME/qairt/2.45.0.260326/lib/python:$PYTHONPATH \
  python research_dev/streamvggt/export_streamvggt_aggregator_qnn.py
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
    """Same patch as the XNNPACK script — eliminate `int(positions.max())+1`."""
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
    def __init__(self, model: StreamVGGT):
        super().__init__()
        self.agg = model.aggregator

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens_list, _patch_start_idx = self.agg(images)
        return tokens_list[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/streamvggt/streamvggt_agg_qnn_fp16.pte")
    ap.add_argument("--soc", default="SM8850",
                    help="SM8850 = OP15 (SD8 Elite Gen 5); SM8650 = OP12 (SD8 Gen 3).")
    args = ap.parse_args()

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )

    # Patch QNN's DecomposeFloorDivide pass: stock version assumes args[1] is
    # an FX node with `.meta["val"]`, but StreamVGGT's `H // patch_size`-style
    # ops produce floor_divide(tensor, int_scalar). Lift int scalars into
    # 0-dim tensors so the inner re-export sees compatible types.
    import torch as _t
    from executorch.backends.qualcomm._passes import decompose_floor_divide as _dfd
    from executorch.backends.qualcomm._passes.utils import merge_decomposed_graph
    from executorch.exir.pass_base import PassResult

    def _patched_call(self, graph_module):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (
                _t.ops.aten.floor_divide.default == node.target
                and not _t.is_floating_point(node.meta["val"])
            ):
                a, b = node.args
                a_val = a.meta["val"] if hasattr(a, "meta") else _t.tensor(a)
                b_val = b.meta["val"] if hasattr(b, "meta") else _t.tensor(b)
                decomposed_module = _t.export.export(
                    _dfd.FloorDivide(), (a_val, b_val), strict=True,
                ).module()
                with graph.inserting_before(node):
                    remap = {"x": a, "y": b}
                    merge_decomposed_graph(
                        remap=remap, target_node=node, target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)

    _dfd.DecomposeFloorDivide.call = _patched_call
    print("  patched DecomposeFloorDivide to handle scalar int args")

    # Patch QnnOperatorSupport.is_node_supported to treat ops without a
    # registered node visitor as unsupported (fall back to CPU) instead of
    # crashing with KeyError. StreamVGGT exercises a handful of these
    # (reciprocal, etc.) — case-by-case allowlisting is whack-a-mole.
    from executorch.backends.qualcomm.partition import qnn_partitioner as _qp
    _orig_is_supported = _qp.QnnOperatorSupport.is_node_supported

    def _safe_is_supported(self, submodules, node):
        try:
            return _orig_is_supported(self, submodules, node)
        except KeyError as e:
            print(f"[QNN Partitioner Op Support]: {node.target.__name__} | "
                  f"NoVisitor → CPU fallback ({e})")
            return False

    _qp.QnnOperatorSupport.is_node_supported = _safe_is_supported
    print("  patched QnnOperatorSupport to fall back on missing visitors")

    _patch_rope_for_export(patch_grid=40)

    print("loading StreamVGGT on cpu (fp32) ...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").cpu().eval()
    # QNN HTP fp16 path: keep fp32 weights here; QNN's HTP fp16 backend will
    # cast at compile time. Avoids the half-precision dropout/contiguous warts.
    wrapper = AggregatorWrapper(model)

    example = torch.zeros(1, 1, 3, 518, 518, dtype=torch.float32)
    print(f"example input: {tuple(example.shape)} {example.dtype}")

    print("\n[1/3] building QNN compile spec (HTP fp16, offline context binary for {})...".format(args.soc))
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    # First attempt used online_prepare=True because offline serialization
    # complained about a too-big context binary. But the on-device DLC parser
    # in QAIRT 2.45 (libQnnModelDlc.so) rejects what ExecuTorch produced —
    # "Failed to open Dlc" — so online_prepare is also a dead end on this
    # device/SDK combination.
    #
    # Switching back to offline context binary, with convert_linear_to_conv2d
    # turned on (passed below to to_edge_transform_and_lower_to_qnn) — this
    # rewrites the heavy nn.Linear ops as Conv2d which often shrinks the QNN
    # graph below the context-binary size limit.
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=backend_options,
    )

    # StreamVGGT's slice_expand_and_flatten lowers to aten.embedding.default
    # whose weight node has no parameter materialised in the lifted program
    # (it's a Parameter sliced via index_select — QNN's op_embedding builder
    # then crashes on a None tensor). Easiest workaround: skip delegating
    # embedding ops; they fall back to CPU. For S=1 there is exactly one such
    # op per camera/register-token expansion, so the runtime cost is trivial.
    # The partitioner compares `node.target.__name__` (a string) against this
    # set, so members must be strings, not OpOverloads.
    skip_node_op_set = {"aten.embedding.default"}

    print("\n[2/3] to_edge + QNN partition + to_executorch ...")
    t0 = time.time()
    et_prog_mgr = to_edge_transform_and_lower_to_qnn(
        wrapper, (example,), compile_spec,
        skip_node_op_set=skip_node_op_set,
        convert_linear_to_conv2d=True,
    )
    print(f"  partitioned in {time.time() - t0:.1f}s")

    et_prog = et_prog_mgr.to_executorch()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        et_prog.write_to_file(f)
    print(f"\n[3/3] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
