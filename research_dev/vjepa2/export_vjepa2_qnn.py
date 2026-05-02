"""
Lower vJEPA2 ViT-L (facebook/vjepa2-vitl-fpc64-256) to ExecuTorch + QNN HTP
for OP15 (SM8850, Hexagon v81). Encoder-only video transformer, 326M params,
24-layer plain ViT. Should export cleanly — none of StreamVGGT's alternating
frame/global attention pattern.

Reuses learnings from research_dev/streamvggt/export_streamvggt_qnn_fp16_clean.py:
  - QNN_TENSOR_TYPE_MAP[torch.float16] missing-entry patch (real ExecuTorch bug)
  - Optional safe-visitor fallback for unmapped ops

Output: a .pte that runs via qnn_executor_runner with the docs-style env
(LD_LIBRARY_PATH=$DEVICE_DIR, ADSP_LIBRARY_PATH=$DEVICE_DIR).
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoModel


MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2EncoderWrapper(nn.Module):
    """Tensor-in, tensor-out wrapper around vJEPA2's encoder.

    Input:  pixel_values_videos (B, C=3, T=64, H=256, W=256) fp16
    Output: last_hidden_state (B, num_tokens, hidden_size=1024)
    """

    def __init__(self, model):
        super().__init__()
        # AutoModel('vjepa2') exposes .encoder which is the Vision Transformer
        # (the predictor head is separate; we don't need it for feature extraction).
        self.encoder = model.encoder

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values_videos)
        # encoder returns BaseModelOutput; we want last_hidden_state.
        return out.last_hidden_state


def _patch_qnn_tensor_type_map() -> None:
    """ExecuTorch's QNN_TENSOR_TYPE_MAP is missing torch.float16. This is
    the real upstream bug we found while trying StreamVGGT."""
    import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
    from executorch.backends.qualcomm.builders import node_visitor as _nv
    if torch.float16 not in _nv.QNN_TENSOR_TYPE_MAP:
        _nv.QNN_TENSOR_TYPE_MAP[torch.float16] = (
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16
        )
        print("  patched QNN_TENSOR_TYPE_MAP: torch.float16 → QNN_DATATYPE_FLOAT_16")


def _patch_qnn_safe_visitor() -> None:
    """Treat ops without a registered QNN visitor as CPU-fallback instead
    of crashing with KeyError. Same logic as in the StreamVGGT script."""
    from executorch.backends.qualcomm.partition import qnn_partitioner as _qp
    _orig = _qp.QnnOperatorSupport.is_node_supported

    def _safe(self, submodules, node):
        try:
            return _orig(self, submodules, node)
        except KeyError as e:
            print(f"[QNN]: {node.target.__name__} | NoVisitor → CPU ({e})")
            return False
    _qp.QnnOperatorSupport.is_node_supported = _safe


def _patch_decompose_floor_divide() -> None:
    """QNN's DecomposeFloorDivide assumes args[1] is an FX node; many
    transformer models do `tensor // int_scalar` which crashes with
    `'int' object has no attribute 'meta'`. Lift int scalars."""
    from executorch.backends.qualcomm._passes import decompose_floor_divide as _dfd
    from executorch.backends.qualcomm._passes.utils import merge_decomposed_graph
    from executorch.exir.pass_base import PassResult

    def _patched_call(self, graph_module):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (
                torch.ops.aten.floor_divide.default == node.target
                and not torch.is_floating_point(node.meta["val"])
            ):
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

    _dfd.DecomposeFloorDivide.call = _patched_call
    print("  patched DecomposeFloorDivide for scalar int args")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/vjepa2/vjepa2_vitl_qnn_fp16.pte")
    ap.add_argument("--soc", default="SM8850")
    ap.add_argument("--frames", type=int, default=64,
                    help="Frames per clip (vJEPA2's standard fpc64).")
    ap.add_argument("--online_prepare", action="store_true",
                    help="Defer compilation to device. Default: try offline first.")
    args = ap.parse_args()

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )

    _patch_qnn_safe_visitor()
    _patch_qnn_tensor_type_map()
    _patch_decompose_floor_divide()

    print(f"\nloading {MODEL_ID} (fp32 cpu) ...")
    model = AutoModel.from_pretrained(MODEL_ID).cpu().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")
    print(f"  config: {model.config.image_size}^2 image, {args.frames} frames, "
          f"hidden={model.config.hidden_size}, layers={model.config.num_hidden_layers}")

    model = model.to(torch.float16)
    wrapper = VJEPA2EncoderWrapper(model)

    # vJEPA2 expects (B, T, C, H, W) — NOT (B, C, T, H, W).
    # The encoder permutes internally before the conv3d patch_embed.
    H = W = model.config.crop_size  # 256
    example = torch.zeros(1, args.frames, 3, H, W, dtype=torch.float16)
    print(f"example input: {tuple(example.shape)} {example.dtype} = "
          f"{example.numel() * example.element_size() / 1e6:.1f} MB")

    print(f"\nbuilding QNN compile spec (HTP fp16, online_prepare={args.online_prepare}, "
          f"soc={args.soc})...")
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=backend_options,
        online_prepare=args.online_prepare,
    )

    # Hook torch.export to cast int64 args of div/mul to fp16 BEFORE any
    # QNN pass runs. Hooking transform_for_export_pipeline runs after
    # DecomposeFloorDivide, which captures dtypes; modifying the graph
    # later trips an _assert_tensor_metadata.
    _orig_export = torch.export.export

    def _patched_export(*a, **kw):
        ep = _orig_export(*a, **kw)
        gm = ep.graph_module if hasattr(ep, "graph_module") else ep
        graph = gm.graph
        casts = 0
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            tname = getattr(node.target, "__name__", str(node.target))
            if tname not in ("aten.div.Tensor", "aten.mul.Tensor",
                             "aten.add.Tensor", "aten.sub.Tensor",
                             "div.Tensor", "mul.Tensor",
                             "add.Tensor", "sub.Tensor"):
                continue
            # Only cast if there's a real int64/fp16 MIX. If all args are
            # int (integer arithmetic), leave alone — the floor_divide
            # decomposition pipeline depends on int dtypes.
            arg_dtypes = []
            for arg in node.args:
                if hasattr(arg, "meta") and "val" in arg.meta:
                    v = arg.meta["val"]
                    arg_dtypes.append(getattr(v, "dtype", None))
                else:
                    arg_dtypes.append(None)
            has_fp = any(d in (torch.float16, torch.float32) for d in arg_dtypes if d is not None)
            has_int64 = any(d == torch.int64 for d in arg_dtypes if d is not None)
            if not (has_fp and has_int64):
                continue
            # Mixed dtype — cast int64 args to fp16.
            new_args = []
            for arg, dt in zip(node.args, arg_dtypes):
                if dt == torch.int64 and hasattr(arg, "meta"):
                    v = arg.meta["val"]
                    with graph.inserting_before(node):
                        cast_node = graph.call_function(
                            torch.ops.aten._to_copy.default,
                            args=(arg,),
                            kwargs={"dtype": torch.float16},
                        )
                        cast_node.meta = dict(arg.meta)
                        cast_node.meta["val"] = v.to(torch.float16)
                    new_args.append(cast_node)
                    casts += 1
                else:
                    new_args.append(arg)
            node.args = tuple(new_args)
        graph.eliminate_dead_code()
        gm.recompile()
        if casts:
            print(f"  cast {casts} int64 args of div/mul/add/sub → fp16 (pre-export)")
        return ep
    torch.export.export = _patched_export

    print("\nto_edge + QNN partition + to_executorch ...")
    t0 = time.time()
    # Skip floor_divide only — adding more ops to the skip set fragments
    # the graph enough that the device-side DLC parser fails to load.
    et_prog_mgr = to_edge_transform_and_lower_to_qnn(
        wrapper, (example,), compile_spec,
        skip_node_op_set={"aten.floor_divide.default"},
    )
    print(f"  partitioned in {time.time() - t0:.1f}s")

    et_prog = et_prog_mgr.to_executorch()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        et_prog.write_to_file(f)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
