#!/usr/bin/env bash
# Confirmed-working harness: qnn-net-run on OP15 retail v81 from shell ADB.
# Targets the QAIRT-shipped InceptionV3 example as a runtime smoke test.
#
# Why this script exists: the standard ExecuTorch QNN path
# (qnn_executor_runner) is blocked on retail v81 because libqnn_executorch_
# backend.so does not call DSPRPC_CONTROL_UNSIGNED_MODULE before opening the
# FastRPC handle. QNN's own libQnnHtp.so does make that call, so qnn-net-run
# works from shell with no root and no signing — exactly the path qblast
# demonstrated in QNNBlast/SUMMARY.md.
#
# Two things the QAIRT-shipped sample script gets wrong on retail OP15:
#   1. ADSP_LIBRARY_PATH only points at the local tmp dir.
#      Fix: include /vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp
#      so libQnnHtpV81Skel can locate cDSP firmware.
#   2. profiling_level off → opaque "Device Creation failure" without context.
#      Fix: --log_level info (and basic profiling) gives composable diagnostics.
#
# Use this as the canonical reference for any future OP15 qnn-net-run launch.

set -euo pipefail
QAIRT="${QAIRT:-$HOME/qairt/2.45.0.260326}"
DEV="${ANDROID_SERIAL:-3C15AU002CL00000}"
NDK="${ANDROID_NDK_ROOT:-$HOME/android/android-ndk-r27c}"
TGT=/data/local/tmp/qnn_netrun_op15

push() { adb -s "$DEV" push "$1" "$TGT/$(basename "$2")" > /dev/null; }

adb -s "$DEV" shell "mkdir -p $TGT $TGT/output"

# Runner + dynamic deps
push "$QAIRT/bin/aarch64-android/qnn-net-run" qnn-net-run
push "$QAIRT/lib/aarch64-android/libQnnHtp.so" libQnnHtp.so
push "$QAIRT/lib/aarch64-android/libQnnHtpPrepare.so" libQnnHtpPrepare.so
push "$QAIRT/lib/aarch64-android/libQnnHtpV81Stub.so" libQnnHtpV81Stub.so
push "$QAIRT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" libQnnHtpV81Skel.so
push "$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" libc++_shared.so

# Model + sample input from the QAIRT InceptionV3 example
MODEL_DIR="$QAIRT/examples/QNN/converter/models"
adb -s "$DEV" push "$MODEL_DIR/input_data_float" "$TGT/" > /dev/null
adb -s "$DEV" push "$MODEL_DIR/input_list_float.txt" "$TGT/" > /dev/null

# Build the model lib if it isn't on disk yet
LIB_OUT=$HOME/qairt/2.45.0.260326/examples/QNN/NetRun/android/model_libs/aarch64-android/libqnn_model_8bit_quantized.so
if [[ ! -f "$LIB_OUT" ]]; then
  echo "Building qnn_model_8bit_quantized.so via qnn-model-lib-generator ..."
  PATH="$NDK:$PATH" "$QAIRT/bin/x86_64-linux-clang/qnn-model-lib-generator" \
    -c "$MODEL_DIR/qnn_model_8bit_quantized.cpp" \
    -b "$MODEL_DIR/qnn_model_8bit_quantized.bin" \
    -o "$HOME/qairt/2.45.0.260326/examples/QNN/NetRun/android/model_libs"
fi
push "$LIB_OUT" libqnn_model_8bit_quantized.so

adb -s "$DEV" shell "
  cd $TGT
  chmod +x qnn-net-run
  LD_LIBRARY_PATH=$TGT \
  ADSP_LIBRARY_PATH=\"$TGT;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\" \
  ./qnn-net-run \
    --model libqnn_model_8bit_quantized.so \
    --input_list input_list_float.txt \
    --backend libQnnHtp.so \
    --output_dir output \
    --profiling_level basic \
    --log_level info
"
