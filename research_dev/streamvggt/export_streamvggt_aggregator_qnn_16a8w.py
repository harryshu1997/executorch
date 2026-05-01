"""
Lower StreamVGGT's aggregator to QNN HTP with 16-bit-act / 8-bit-weight
quantization (PT2E), targeting OP15 (SM8850, Hexagon v81). This is the path
that should fit the offline context binary — fp16 export hit Error 30002
(graph too big to serialize), and online_prepare DLC didn't load on device.

Mirrors the official whisper.py quant flow:
  torch.export -> prepare_pt2e -> calibrate -> convert_pt2e -> lower_to_qnn

Calibration data: 3 kitchen frames from VGGT's example set, preprocessed at
518² fp32 — good enough to populate the MinMaxObserver ranges.
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
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402


def _patch_rope_with_precomputed_buffers(model, patch_grid: int = 40,
                                          feature_dim: int = 32) -> None:
    """Register cos_comp/sin_comp as real torch buffers on the rope module,
    replace _compute_frequency_components to return them. This collapses
    the rope's runtime arange/pow/reciprocal/cos/sin chain (which produced
    CPU-fallback partitions for `aten.reciprocal.default` and
    `aten.pow.Scalar`) into a single buffer lookup that QNN handles as a
    standard embedding/gather op.

    Underlying root cause from QNN compile log:
        <E> No graph inputs present for graph [0]
    happens when CPU↔QNN bounces shred the graph so QNN's piece has no
    input source. Eliminating those bounces fixes the serializer.
    """
    rope = model.aggregator.rope
    # Use the rope's own _compute_frequency_components to avoid drift in the
    # arange/einsum recipe. Feature_dim is half the token's last dim
    # (rope splits tokens in half for vertical vs horizontal axes).
    cos_buf, sin_buf = rope._compute_frequency_components(
        feature_dim, patch_grid, torch.device("cpu"), torch.float32,
    )
    cos_buf = cos_buf.contiguous()
    sin_buf = sin_buf.contiguous()
    # Register as buffers so torch.export lifts them as parameters and
    # QNN's op_embedding builder can look them up via get_parameter.
    rope.register_buffer("_cos_buf", cos_buf)
    rope.register_buffer("_sin_buf", sin_buf)

    def _patched_compute_freq(self, dim, seq_len, device, dtype):
        # Trim/expand the precomputed buffers to (seq_len, dim*2). For the
        # static-shape case (seq_len <= patch_grid), this is a slice.
        return self._cos_buf[:seq_len, :dim * 2].to(dtype), \
               self._sin_buf[:seq_len, :dim * 2].to(dtype)
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
    print(f"  rope: cos/sin buffers registered ({tuple(cos_buf.shape)}), "
          f"reciprocal/pow chain eliminated")


def _patch_executorch_qnn_passes() -> None:
    """Same fixes as the fp16 export so partition succeeds on StreamVGGT."""
    from executorch.backends.qualcomm._passes import decompose_floor_divide as _dfd
    from executorch.backends.qualcomm._passes.utils import merge_decomposed_graph
    from executorch.exir.pass_base import PassResult
    from executorch.backends.qualcomm.partition import qnn_partitioner as _qp

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

    # QNN's RemoveRedundancy pass already targets _to_dim_order_copy but
    # only deletes them when dtypes match. After 16a8w quant the dtypes
    # differ at boundaries, so they survive — and QNN's validator then
    # rejects them with 0xc26. Force the condition to always return True
    # so they're stripped regardless. The "memory format change" they
    # represent is a no-op for HTP which uses its own native layout.
    from executorch.backends.qualcomm._passes import remove_redundancy as _rr
    # The dim_order_copy ops in StreamVGGT split into 3 classes:
    #   (a) layout-only, same dtype both ends → safe to strip (default does this)
    #   (b) bool→fp32 mask conversion → must keep (load-bearing)
    #   (c) fp32↔int8 boundary (PT2E quant artefact) → safe to strip; the
    #       real data is already cast by the surrounding quant/dequant ops
    #
    # Default condition strips only (a). The (c) ops survive and trigger
    # QNN's 0xc26. Force-strip everything that isn't a bool transition.
    def _smart_strip(self, node):
        out_dt = node.meta["val"].dtype
        in_dt = node.args[0].meta["val"].dtype
        if out_dt == torch.bool or in_dt == torch.bool:
            return False  # mask conversion — keep
        return True  # layout-only or quant boundary — strip
    _rr.RemoveRedundancy._dim_order_op_condition = _smart_strip
    print("  patched RemoveRedundancy: strip dim_order_copy except bool transitions")

    _orig_supp = _qp.QnnOperatorSupport.is_node_supported

    def _safe_supp(self, submodules, node):
        try:
            return _orig_supp(self, submodules, node)
        except KeyError as e:
            print(f"[QNN]: {node.target.__name__} | NoVisitor → CPU ({e})")
            return False
    _qp.QnnOperatorSupport.is_node_supported = _safe_supp
    print("  patched DecomposeFloorDivide + safe-visitor fallback")


class AggregatorWrapper(nn.Module):
    def __init__(self, model: StreamVGGT):
        super().__init__()
        self.agg = model.aggregator

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens_list, _ = self.agg(images)
        return tokens_list[-1]


def get_calibration_inputs(n: int = 3) -> list[tuple[torch.Tensor]]:
    img_dir = VGGT_ROOT / "examples" / "kitchen" / "images"
    paths = sorted(img_dir.glob("*.png"))[:n]
    out = []
    for p in paths:
        imgs, _ = load_and_preprocess_images_square([str(p)], target_size=518)
        out.append((imgs.unsqueeze(0).float().contiguous(),))
    print(f"  loaded {len(out)} calibration frames from {img_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/streamvggt/streamvggt_agg_qnn_16a8w_half.pte")
    ap.add_argument("--soc", default="SM8850")
    ap.add_argument("--n_calib", type=int, default=3)
    args = ap.parse_args()

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.examples.qualcomm.utils import make_quantizer
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )
    from executorch.backends.qualcomm._passes.qnn_pass_manager import (
        get_capture_program_passes,
    )
    from torchao.quantization.pt2e import MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    _patch_executorch_qnn_passes()

    print("\nloading StreamVGGT (fp32 cpu) ...")
    model = StreamVGGT.from_pretrained("lch01/StreamVGGT").cpu().eval()
    print(f"  aa_block_num: {model.aggregator.aa_block_num} (full depth)")
    _patch_rope_with_precomputed_buffers(model, patch_grid=40, feature_dim=32)
    wrapper = AggregatorWrapper(model)

    example = torch.zeros(1, 1, 3, 518, 518, dtype=torch.float32)
    print(f"example input: {tuple(example.shape)} {example.dtype}")

    print("\n[1/5] torch.export to module ...")
    t0 = time.time()
    with torch.no_grad():
        exported = torch.export.export(wrapper, (example,), strict=True).module()
    print(f"  exported in {time.time()-t0:.1f}s")

    soc = getattr(QcomChipset, args.soc)
    print(f"\n[2/5] make_quantizer (16a8w, per-channel linear+conv, soc={args.soc}) ...")
    # make_quantizer takes soc_model as a string ("SM8850"), not the enum.
    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
        eps=2**-20,
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=args.soc,
    )

    print("\n[3/5] prepare_pt2e + calibrate ...")
    t0 = time.time()
    with torch.no_grad():
        exported = prepare_pt2e(exported, quantizer)
        cal = get_calibration_inputs(args.n_calib)
        for i, inp in enumerate(cal):
            print(f"  calib frame {i+1}/{len(cal)}...", flush=True)
            exported(*inp)
        exported = convert_pt2e(exported)
    print(f"  prepare+calibrate+convert in {time.time()-t0:.1f}s")

    # Diagnostic: confirm quantization actually applied. After convert_pt2e
    # we should see quantize_per_tensor/per_channel + dequantize ops in the
    # graph if PT2E worked. If counts are 0, the quantizer didn't annotate.
    q_count = dq_count = 0
    op_hist: dict[str, int] = {}
    for node in exported.graph.nodes:
        if node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            op_hist[name] = op_hist.get(name, 0) + 1
            if "quantize" in name and "dequantize" not in name:
                q_count += 1
            elif "dequantize" in name:
                dq_count += 1
    print(f"  diagnostic: quantize ops = {q_count}, dequantize ops = {dq_count}")
    print(f"  top 10 ops in converted graph:")
    for name, n in sorted(op_hist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {n:>4}  {name}")

    print(f"\n[4/5] lower_to_qnn (HTP, offline context binary) ...")
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=soc,
        backend_options=backend_options,
    )
    # No skips — let everything attempt QNN. With the rope cos/sin buffers
    # registered, embedding gets a real param weight, and we want to stop
    # fragmenting the graph with skip-driven CPU boundaries (those produce
    # "No graph inputs present for graph [0]" inside QNN's serializer).
    skip_node_op_set = set()
    t0 = time.time()
    # convert_linear_to_conv2d=True generated _to_dim_order_copy ops QNN
    # can't validate (24 layer-boundary failures). Drop it; 16a8w alone
    # halves the weight size which should be enough for the context binary.
    if q_count == 0:
        print("\n=== ABORT: quant didn't apply ===")
        return

    # Remove dim_order_copy nodes from the graph before partitioning. They
    # are memory-format hints (no real compute), but QNN 2.45's validator
    # rejects them with 0xc26, and skipping them to CPU produces input-less
    # QNN partitions ("No graph inputs present for graph [0]" 0x7532). The
    # safer move: drop them outright. Each call replaces the node with a
    # passthrough of its single tensor input.
    def _strip_dim_order_copy(gm: torch.fx.GraphModule):
        graph = gm.graph
        removed = 0
        # Match by op name suffix since target may be EdgeOpOverload.
        for node in list(graph.nodes):
            tname = getattr(node.target, "__name__", str(node.target))
            if tname == "dim_order_ops._to_dim_order_copy.default":
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                removed += 1
        graph.eliminate_dead_code()
        gm.recompile()
        if removed:
            print(f"  stripped {removed} dim_order_copy nodes from edge graph")
        return gm

    # Hook the strip pass into to_edge_transform_and_lower_to_qnn via a
    # monkey-patch of the partitioner factory's input. Easiest: do it after
    # to_edge_transform_and_lower runs the IR conversion but before partition.
    # The cleanest seam ExecuTorch exposes is a Pass run on the edge program.
    import executorch.exir.program._program as _prog_mod
    _orig_to_edge = _prog_mod.to_edge_transform_and_lower

    def _wrapped_to_edge(*a, **kw):
        # Strip dim_order_copy on each method's graph after conversion to
        # edge IR but before partitioning. We re-implement the inner sequence.
        return _orig_to_edge(*a, **kw)

    # Simpler tactic: pre-strip the exported module's graph BEFORE
    # to_edge_transform_and_lower_to_qnn. Edge-IR conversion may reinsert
    # them, but at minimum we eliminate any pre-IR copies torch.export
    # produced from .contiguous() calls in StreamVGGT.
    _strip_dim_order_copy(exported)

    et_prog_mgr = to_edge_transform_and_lower_to_qnn(
        exported, (example,), compile_spec,
        skip_node_op_set=skip_node_op_set,
        passes_job=get_capture_program_passes(),
        convert_linear_to_conv2d=True,
    )
    print(f"  lowered in {time.time()-t0:.1f}s")

    et_prog = et_prog_mgr.to_executorch()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        et_prog.write_to_file(f)
    print(f"\n[5/5] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
