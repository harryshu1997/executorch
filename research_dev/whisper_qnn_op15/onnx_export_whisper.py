"""
ONNX-export Whisper-Tiny encoder for the QNN-native pipeline:
  ONNX -> qnn-onnx-converter -> libModel.so -> qnn-net-run on OP15.

Why not use the .pte we already produced? Because ExecuTorch's QNN backend
skips the DSPRPC_CONTROL_UNSIGNED_MODULE call that retail OP15 firmware
requires for shell-context FastRPC. QNN's own runner (qnn-net-run) makes
that call inside libQnnHtp.so, which is why it works on retail v81 from
shell while the executor_runner does not.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq


class WhisperEncoder(torch.nn.Module):
    def __init__(self, model_id: str = "openai/whisper-tiny"):
        super().__init__()
        full = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).eval()
        self.encoder = full.model.encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 80, 3000) fp32
        return self.encoder(mel).last_hidden_state  # (B, 1500, 384)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_dev/whisper_qnn_op15/whisper_enc.onnx")
    args = ap.parse_args()

    print("loading whisper-tiny encoder...")
    m = WhisperEncoder().eval()

    example = torch.zeros(1, 80, 3000, dtype=torch.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"exporting ONNX → {out} ...")
    torch.onnx.export(
        m, (example,), str(out),
        input_names=["mel"],
        output_names=["features"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,  # static shape
    )
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    # Save the example input as raw bytes for qnn-net-run.
    raw = out.with_suffix(".raw")
    example.numpy().tofile(raw)
    print(f"  wrote {raw} ({raw.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
