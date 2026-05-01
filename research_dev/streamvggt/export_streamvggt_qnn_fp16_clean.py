"""
v14 attempt: combine the best of every previous experiment in fp16 (no PT2E
quant) — that combo we haven't tried yet.

  - fp16 weights/inputs (no PT2E)         → eliminates dtype-mismatch class
                                            (`Tensor 0x32 != 0x232`)
  - rope cos/sin precomputed as buffers   → eliminates rope reciprocal/pow
                                            CPU fallback partitions
  - smart-strip dim_order_copy ops        → eliminates 0xc26 validator
                                            failures except bool→fp32 masks
  - online_prepare=True (DLC format)      → bypasses offline-context-binary
                                            size assertion
  - no skip_node_op_set (single QNN partition)

If the resulting .pte loads on OP15 (the previous online_prepare DLC didn't
load — but that was without the rope/dim_order_copy fixes; the device-side
DLC parser might accept the cleaner graph).
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


def _patch_rope_with_buffers(model, patch_grid: int = 40, feature_dim: int = 32) -> None:
    rope = model.aggregator.rope
    cos_buf, sin_buf = rope._compute_frequency_components(
        feature_dim, patch_grid, torch.device("cpu"), torch.float16,
    )
    rope.register_buffer("_cos_buf", cos_buf.to(torch.float16).contiguous())
    rope.register_buffer("_sin_buf", sin_buf.to(torch.float16).contiguous())

    def _patched_compute_freq(self, dim, seq_len, device, dtype):
        # No slicing — buffers are pre-shaped to (patch_grid, feature_dim).
        # The earlier slice `[:seq_len, :dim]` was a no-op tensor view that
        # may have been causing downstream shape-inference issues.
        return self._cos_buf, self._sin_buf
    _rope.RotaryPositionEmbedding2D._compute_frequency_components = _patched_compute_freq

    def _exportable_forward(self, tokens, positions):
        fd = tokens.size(-1) // 2
        cos_comp, sin_comp = self._compute_frequency_components(
            fd, patch_grid, tokens.device, tokens.dtype
        )
        v, h = tokens.chunk(2, dim=-1)
        v = self._apply_1d_rope(v, positions[..., 0], cos_comp, sin_comp)
        h = self._apply_1d_rope(h, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((v, h), dim=-1)
    _rope.RotaryPositionEmbedding2D.forward = _exportable_forward
    print(f"  rope: cos/sin buffers registered ({tuple(cos_buf.shape)})")


def _patch_qnn_passes() -> None:
    import torch as _t
    from executorch.backends.qualcomm._passes import (
        decompose_floor_divide as _dfd,
        remove_redundancy as _rr,
    )
    from executorch.backends.qualcomm._passes.utils import merge_decomposed_graph
    from executorch.exir.pass_base import PassResult
    from executorch.backends.qualcomm.partition import qnn_partitioner as _qp

    def _patched_floor_div(self, graph_module):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (_t.ops.aten.floor_divide.default == node.target
                    and not _t.is_floating_point(node.meta["val"])):
                a, b = node.args
                a_val = a.meta["val"] if hasattr(a, "meta") else _t.tensor(a)
                b_val = b.meta["val"] if hasattr(b, "meta") else _t.tensor(b)
                decomposed = _t.export.export(
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

    # Restore smart-strip (v15 baseline). v16 default-only led to OOM /
    # silent crash during online_prepare with smaller planned buffers
    # (7 MB vs v15's 259 MB), suggesting too many CPU partitions.
    def _smart_strip(self, node):
        out_dt = node.meta["val"].dtype
        in_dt = node.args[0].meta["val"].dtype
        if out_dt == _t.bool or in_dt == _t.bool:
            return False
        return True
    _rr.RemoveRedundancy._dim_order_op_condition = _smart_strip

    _orig_supp = _qp.QnnOperatorSupport.is_node_supported
    def _safe_supp(self, submodules, node):
        try:
            return _orig_supp(self, submodules, node)
        except KeyError as e:
            print(f"[QNN]: {node.target.__name__} | NoVisitor → CPU ({e})")
            return False
    _qp.QnnOperatorSupport.is_node_supported = _safe_supp
    print("  patched DecomposeFloorDivide + smart-strip dim_order_copy + safe-visitor")


class AggregatorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.agg = model.aggregator

    def forward(self, images):
        # Aggressive fusion disruption: take 4D (B*S, C, H, W) input,
        # then unsqueeze to (B, S, C, H, W) inside the wrapper. This
        # changes the input boundary shape so QNN can't conflate it
        # with patch_embed's view_copy of the conv weight.
        images = images.unsqueeze(0)  # (1, C, H, W) → (1, 1, C, H, W)
        tokens_list, _ = self.agg(images)
        return tokens_list[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/streamvggt/streamvggt_agg_qnn_fp16_v14.pte")
    ap.add_argument("--soc", default="SM8850")
    args = ap.parse_args()

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )

    _patch_qnn_passes()

    # QNN_TENSOR_TYPE_MAP in ExecuTorch's QNN backend is missing
    # torch.float16. Adding it so fp16 tensors lower instead of throwing
    # KeyError('torch.float16') from inside the op visitors. The QNN SDK
    # itself does support FLOAT_16; this is purely a missing dict entry
    # in ExecuTorch's wrapper.
    import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
    from executorch.backends.qualcomm.builders import node_visitor as _nv
    _nv.QNN_TENSOR_TYPE_MAP[torch.float16] = (
        PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16
    )
    print("  patched QNN_TENSOR_TYPE_MAP: torch.float16 → QNN_DATATYPE_FLOAT_16")

    print("\nloading StreamVGGT (fp32 cpu) ...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").cpu().eval()
    _patch_rope_with_buffers(model, patch_grid=40, feature_dim=32)
    # v26: try fp32 throughout — no mixed-precision issues. The QNN
    # validator was rejecting bmm with bf16/fp16 mismatch (codes 0x232
    # vs 0x216). With fp32 everywhere, no QNN-internal bf16 promotion
    # can cause a mismatch.
    wrapper = AggregatorWrapper(model)

    example = torch.zeros(1, 3, 518, 518, dtype=torch.float32)
    print(f"example input: {tuple(example.shape)} {example.dtype}")

    print(f"\nbuilding QNN compile spec (HTP fp16, online_prepare=True for {args.soc})...")
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=backend_options,
        online_prepare=True,
    )

    # The DecomposeScaledDotProductAttention pass produces _safe_softmax
    # nodes that promote to fp32, leaking into downstream CPU bmm. Wrap
    # to_edge_transform_and_lower_to_qnn so we can intercept the post-
    # decomp graph and rewrite _safe_softmax → softmax (preserves dtype).
    from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
    _orig_transform = QnnPassManager.transform_for_export_pipeline

    def _patched_transform(self, exported_program, *a, **kw):
        ep = _orig_transform(self, exported_program, *a, **kw)
        gm = ep.graph_module if hasattr(ep, "graph_module") else ep
        graph = gm.graph

        # 1) Rewrite _safe_softmax → _softmax (preserves input dtype)
        replaced = 0
        for node in list(graph.nodes):
            tname = getattr(node.target, "__name__", str(node.target))
            if tname in ("_safe_softmax.default", "aten._safe_softmax.default"):
                with graph.inserting_after(node):
                    new_node = graph.call_function(
                        torch.ops.aten._softmax.default,
                        args=(node.args[0], node.args[1], False),
                        kwargs={},
                    )
                    new_node.meta = dict(node.meta)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
                replaced += 1

        # 2) Cast every fp32 OR bf16 constant tensor to fp16. The CPU bmm
        #    fails with dtype mismatch where one tensor is bf16 (QNN code
        #    0x232) and another is fp16 (0x216). Walk get_attr nodes.
        const_cast = 0
        for node in list(graph.nodes):
            if node.op == "get_attr":
                v = getattr(gm, node.target, None)
                if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.bfloat16):
                    setattr(gm, node.target, v.to(torch.float16))
                    if "val" in node.meta and hasattr(node.meta["val"], "dtype"):
                        node.meta["val"] = node.meta["val"].to(torch.float16)
                    const_cast += 1

        # 3) For every node whose meta says fp32 OR bf16 output, cast its
        #    dtype metadata to fp16.
        meta_fix = 0
        for node in list(graph.nodes):
            if node.op == "call_function" and "val" in node.meta:
                v = node.meta["val"]
                if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.bfloat16):
                    try:
                        node.meta["val"] = v.to(torch.float16)
                        meta_fix += 1
                    except Exception:
                        pass

        # 4) Insert explicit aten._to_copy(dtype=fp16) after any node
        #    whose actual computation might produce bf16 at runtime.
        #    Specifically targets aten.to.dtype and dtype-changing _to_copy
        #    nodes that promote to bf16.
        bf16_kills = 0
        for node in list(graph.nodes):
            if node.op == "call_function":
                # Match aten._to_copy.default with dtype kwarg = bf16
                if "kwargs" in dir(node) and node.kwargs.get("dtype") == torch.bfloat16:
                    new_kwargs = {**node.kwargs, "dtype": torch.float16}
                    node.kwargs = new_kwargs
                    bf16_kills += 1
                # Match aten.to(*, dtype=bf16)
                if (len(node.args) >= 2 and node.args[-1] == torch.bfloat16):
                    node.args = (*node.args[:-1], torch.float16)
                    bf16_kills += 1

        graph.eliminate_dead_code()
        gm.recompile()
        if replaced or const_cast or meta_fix or bf16_kills:
            print(f"  rewrote {replaced} _safe_softmax, cast {const_cast} const, "
                  f"fixed {meta_fix} metas, killed {bf16_kills} bf16 ops → fp16")
        return ep
    QnnPassManager.transform_for_export_pipeline = _patched_transform

    print("\nto_edge + QNN partition + to_executorch ...")
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
