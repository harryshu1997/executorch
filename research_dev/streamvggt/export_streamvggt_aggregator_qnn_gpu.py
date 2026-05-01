"""
Lower StreamVGGT's aggregator to ExecuTorch + QNN GPU (Adreno) for OP15.
Parallels the HTP 16a8w script but uses the kGpuBackend, which:
  - takes fp16 inputs/weights directly (no PT2E quant calibration)
  - runs OpenCL kernels on the Adreno GPU
  - has DIFFERENT op-support coverage than HTP, in particular it does
    not enforce the dim_order_copy validator that's been blocking the
    HTP path with error 0xc26.

Slower than NPU but proven path. whisper.py's --backend gpu flow is the
template. Online_prepare must be True for GPU per executorch QNN docs.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

STREAMVGGT_ROOT = Path("/home/myid/zs89458/Documents/StreamVGGT")
VGGT_ROOT = Path("/home/myid/zs89458/Documents/vggt")
sys.path.insert(0, str(STREAMVGGT_ROOT / "src"))
sys.path.insert(0, str(VGGT_ROOT))

from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from streamvggt.layers import rope as _rope  # noqa: E402


def _patch_rope_for_export(patch_grid: int = 40) -> None:
    def exportable_forward(self, tokens, positions):
        feature_dim = tokens.size(-1) // 2
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, patch_grid, tokens.device, tokens.dtype
        )
        v, h = tokens.chunk(2, dim=-1)
        v = self._apply_1d_rope(v, positions[..., 0], cos_comp, sin_comp)
        h = self._apply_1d_rope(h, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((v, h), dim=-1)
    _rope.RotaryPositionEmbedding2D.forward = exportable_forward
    print(f"  patched rope.forward: max_position={patch_grid}")


def _patch_qnn_safe_visitor() -> None:
    """Same safe-visitor fallback as HTP path: missing op visitors → CPU."""
    from executorch.backends.qualcomm.partition import qnn_partitioner as _qp
    _orig = _qp.QnnOperatorSupport.is_node_supported

    def _safe(self, submodules, node):
        try:
            return _orig(self, submodules, node)
        except KeyError as e:
            print(f"[QNN]: {node.target.__name__} | NoVisitor → CPU ({e})")
            return False
    _qp.QnnOperatorSupport.is_node_supported = _safe

    # DecomposeFloorDivide patch for scalar args (same as HTP).
    from executorch.backends.qualcomm._passes import decompose_floor_divide as _dfd
    from executorch.backends.qualcomm._passes.utils import merge_decomposed_graph
    from executorch.exir.pass_base import PassResult

    def _patched_floor_div(self, graph_module):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (torch.ops.aten.floor_divide.default == node.target
                    and not torch.is_floating_point(node.meta["val"])):
                a, b = node.args
                a_val = a.meta["val"] if hasattr(a, "meta") else torch.tensor(a)
                b_val = b.meta["val"] if hasattr(b, "meta") else torch.tensor(b)
                decomposed = torch.export.export(
                    _dfd.FloorDivide(), (a_val, b_val), strict=True,
                ).module()
                with graph.inserting_before(node):
                    remap = {"x": a, "y": b}
                    merge_decomposed_graph(
                        remap=remap, target_node=node, target_graph=graph,
                        decomposed_graph_module=decomposed,
                    )
                    graph.erase_node(node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
    _dfd.DecomposeFloorDivide.call = _patched_floor_div


class AggregatorWrapper(nn.Module):
    def __init__(self, model: StreamVGGT):
        super().__init__()
        self.agg = model.aggregator

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens_list, _ = self.agg(images)
        return tokens_list[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/streamvggt/streamvggt_agg_qnn_gpu_fp16.pte")
    ap.add_argument("--soc", default="SM8850")
    args = ap.parse_args()

    from executorch.backends.qualcomm.serialization.qc_schema import (
        QcomChipset,
        QnnExecuTorchBackendType,
        QnnExecuTorchGpuPrecision,
    )
    from executorch.backends.qualcomm.utils.utils import (
        generate_gpu_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )

    _patch_rope_for_export(patch_grid=40)
    _patch_qnn_safe_visitor()

    print("\nloading StreamVGGT (fp16 cpu) ...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").cpu().eval()
    model = model.to(torch.float16)
    wrapper = AggregatorWrapper(model)

    example = torch.zeros(1, 1, 3, 518, 518, dtype=torch.float16)
    print(f"example input: {tuple(example.shape)} {example.dtype}")

    print(f"\nbuilding QNN compile spec (GPU fp16, online_prepare=True for {args.soc})...")
    backend_options = generate_gpu_compiler_spec(
        precision=QnnExecuTorchGpuPrecision.kGpuPrecisionFp16,
    )
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=backend_options,
        online_prepare=True,  # GPU backend requires online_prepare per ET docs
    )

    print("\nto_edge + QNN GPU partition + to_executorch ...")
    t0 = time.time()
    et_prog_mgr = to_edge_transform_and_lower_to_qnn(
        wrapper, (example,), compile_spec,
    )
    print(f"  partitioned in {time.time()-t0:.1f}s")

    et_prog = et_prog_mgr.to_executorch()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        et_prog.write_to_file(f)
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
