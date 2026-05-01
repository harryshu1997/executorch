"""
Derive a multi-modal task-arrival trace from an AR/VR recording.

Input:  any video file with an audio track. For Aria VRS files, set
        --aria and we'll use projectaria_tools to extract the ego camera
        stream + audio. Everything else (VAD, intent detection) is
        format-agnostic so this same script drives both paths.

Output: a timestamped JSON trace in the schema MULTI_MODEL_PLAN §3c:
        {"t": <seconds>, "task": <str>, "deadline_s": <float>,
         "priority": "foreground"|"background", ... task-specific fields}

Triggers:
  - VGGT_encoder  : periodic at --vggt_fps (default 2 fps).
  - Whisper       : per utterance, via Silero VAD on the audio stream.
  - LLM_query     : per utterance whose (stub) transcript contains an
                    intent keyword (question words / "hey assistant").
                    In a later pass this is replaced by actually running
                    Whisper on the utterance clip and scanning real text.

Why this lives in the harness and not a library:
The derivation is a *research question* (what counts as an LLM event?).
Keep it explicit and editable, not hidden in a framework.
"""
import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TaskEvent:
    t: float            # seconds since session start
    task: str           # "VGGT_encoder" | "Whisper" | "LLM_query"
    deadline_s: float
    priority: str       # "foreground" | "background"
    payload: dict       # task-specific (frame index, utterance clip, transcript)


# ------------------------------------------------------------------------
# Input: video + audio extraction. One backend for Aria, one for plain mp4.
# ------------------------------------------------------------------------

def read_video_metadata(path: Path) -> dict:
    """Return {duration_s, fps, has_audio, audio_sr} for a standard video."""
    import imageio.v3 as iio
    meta = iio.immeta(str(path), plugin="pyav")
    # imageio returns fps; duration often via another call
    dur = meta.get("duration", None)
    if dur is None:
        # fallback: count frames with iterator (slow but reliable)
        n = sum(1 for _ in iio.imiter(str(path), plugin="pyav"))
        dur = n / float(meta.get("fps", 30.0))
    return {
        "duration_s": float(dur),
        "fps": float(meta.get("fps", 30.0)),
        "audio_fps": float(meta.get("audio_fps") or 0.0),
    }


def extract_audio_wav(path: Path, out_wav: Path, target_sr: int = 16000) -> bool:
    """Decode the audio track to a 16 kHz mono WAV. Returns False if no audio."""
    # Use imageio/pyav via a small shim. ffmpeg would be cleaner but we want
    # zero system deps; if imageio-ffmpeg is present, it can dump audio.
    try:
        import av  # PyAV (pulled in by imageio-ffmpeg via pyav plugin)
    except ImportError:
        raise RuntimeError("PyAV required for audio extraction: pip install av")
    container = av.open(str(path))
    astreams = [s for s in container.streams if s.type == "audio"]
    if not astreams:
        container.close()
        return False
    import numpy as np
    import wave
    a = astreams[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks = []
    for packet in container.demux(a):
        for frame in packet.decode():
            for rf in resampler.resample(frame) or ():
                chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return False
    pcm = np.concatenate(chunks).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(target_sr)
        w.writeframes(pcm.tobytes())
    return True


def read_aria_vrs(path: Path, cache_dir: Path) -> dict:
    """Pull ego-video + audio from an Aria VRS file into a temp dir so the
    rest of the pipeline treats it like any other recording."""
    try:
        from projectaria_tools.core import data_provider, mps  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "projectaria_tools not installed. `pip install projectaria-tools` "
            "or run on a non-Aria video and drop --aria."
        )
    raise NotImplementedError(
        "Aria VRS extraction: fill in using projectaria_tools once the dataset "
        "is downloaded. Stub here so the rest of the pipeline runs today."
    )


# ------------------------------------------------------------------------
# Whisper-trigger detection: VAD on audio.
# ------------------------------------------------------------------------

def vad_utterances(wav_path: Path, sample_rate: int = 16000) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) speech segments via Silero VAD."""
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            trust_repo=True, verbose=False,
        )
        get_speech_timestamps = utils[0]
    except Exception as e:
        raise RuntimeError(f"Silero VAD unavailable: {e}")
    import wave, numpy as np
    with wave.open(str(wav_path), "rb") as w:
        assert w.getframerate() == sample_rate, f"expected {sample_rate} Hz"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = torch.from_numpy(pcm.astype("float32") / 32768.0)
    ts = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
    return [(t["start"] / sample_rate, t["end"] / sample_rate) for t in ts]


# ------------------------------------------------------------------------
# LLM-trigger detection: intent match on transcript.
#   For v1, we skip actually running Whisper and treat *every* utterance
#   longer than some duration as a potential query. When we wire up
#   Whisper we'll replace this with transcript-based matching.
# ------------------------------------------------------------------------

INTENT_KEYWORDS = (
    # Wake-words (real AR-assistant triggers)
    "hey assistant", "hey aria", "okay glasses", "hey glasses",
    # Question heads (casual-speech question-like utterances)
    "what ", "what's", "whats",
    "how ", "how's", "hows",
    "where ", "when ", "why ", "who ", "which ",
    "is it", "is this", "is that", "is there", "are you", "are there",
    "do you", "don't you", "can you", "could you", "would you", "did you",
    # Action intents
    "show me", "find", "remind me", "set timer", "set a timer", "call",
    "tell me", "recommend",
)


def load_aria_diarization(csv_path: Path) -> list[tuple[float, float, str, str]]:
    """Read Aria's diarization CSV into (start_s, end_s, speaker, transcript)
    tuples, relativized so the earliest utterance starts at t=0."""
    import csv
    rows: list[tuple[int, int, str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["start_timestamp_ns"]), int(r["end_timestamp_ns"]),
                         r["speaker"], r["content"]))
    if not rows:
        return []
    t0 = min(r[0] for r in rows)
    return [((s - t0) / 1e9, (e - t0) / 1e9, sp, c) for (s, e, sp, c) in rows]


def is_intent_utterance(duration_s: float, transcript: str | None) -> bool:
    """True if an utterance looks like a user query that should trigger LLM."""
    if transcript is not None:
        text = transcript.lower()
        return any(k in text for k in INTENT_KEYWORDS)
    # v1 heuristic: any utterance longer than 0.8s is candidate. This
    # overestimates LLM triggers but is fine for a scheduler stress test.
    return duration_s >= 0.8


# ------------------------------------------------------------------------
# Assembly.
# ------------------------------------------------------------------------

def build_trace(
    duration_s: float,
    vggt_fps: float,
    utterances: list[tuple[float, float, str | None, str | None]],
) -> Iterator[TaskEvent]:
    """utterances: list of (start_s, end_s, speaker, transcript). speaker
    and transcript may be None (VAD path) or populated (Aria diarization)."""
    # VGGT: periodic
    step = 1.0 / vggt_fps
    t = 0.0
    i = 0
    while t < duration_s:
        yield TaskEvent(
            t=t, task="VGGT_encoder",
            deadline_s=step * 0.9,
            priority="background",
            payload={"frame_idx": i},
        )
        t += step
        i += 1

    # Whisper + LLM: one per utterance. Only SELF utterances (the wearer
    # speaking) plausibly trigger a local Whisper; OTHER utterances would
    # still need recognition in a real system, so include both but mark
    # the speaker in payload.
    for i, (start, end, speaker, transcript) in enumerate(utterances):
        yield TaskEvent(
            t=start, task="Whisper",
            deadline_s=0.5, priority="foreground",
            payload={"utt_idx": i, "clip_start_s": start, "clip_end_s": end,
                     "speaker": speaker, "transcript": transcript},
        )
        if is_intent_utterance(end - start, transcript):
            yield TaskEvent(
                t=end, task="LLM_query",
                deadline_s=2.0, priority="foreground",
                payload={"utt_idx": i, "transcript": transcript, "speaker": speaker},
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video/VRS file.")
    ap.add_argument("--aria", action="store_true", help="Treat input as Aria VRS.")
    ap.add_argument("--diarization", default=None,
                    help="Path to Aria diarization_results.csv. If set, use that instead of VAD.")
    ap.add_argument("--vggt_fps", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="Output JSON trace path.")
    ap.add_argument("--cache", default="/tmp/trace_cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)

    if args.aria:
        meta = read_aria_vrs(video, cache)   # not implemented yet
        wav = cache / "audio.wav"
    else:
        meta = read_video_metadata(video)
        wav = cache / (video.stem + ".wav")
        if not extract_audio_wav(video, wav):
            print(f"WARNING: {video} has no audio track; Whisper/LLM events skipped.")
            wav = None

    print(f"video duration: {meta['duration_s']:.2f} s")

    if args.diarization:
        diar = load_aria_diarization(Path(args.diarization))
        # For alignment: Aria preview mp4 typically starts at or near the
        # first utterance's SLAM-aligned time. We relativized to first
        # utterance = t=0 so video t=0 === diarization t=0. Caveat: if the
        # video starts significantly *before* the first utterance, the
        # first few seconds will have no voice events, which is fine.
        # Trim to video duration.
        diar = [u for u in diar if u[0] < meta["duration_s"]]
        utterances = [(s, min(e, meta["duration_s"]), sp, c) for (s, e, sp, c) in diar]
        n_self = sum(1 for u in utterances if u[2] == "SELF")
        n_other = sum(1 for u in utterances if u[2] == "OTHER")
        print(f"utterances (Aria diarization): {len(utterances)}  (SELF={n_self}, OTHER={n_other})")
    else:
        vad = vad_utterances(wav) if wav else []
        utterances = [(s, e, None, None) for (s, e) in vad]
        print(f"utterances (VAD): {len(utterances)}")

    trace = list(build_trace(meta["duration_s"], args.vggt_fps, utterances))
    trace.sort(key=lambda e: (e.t, e.task))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ev in trace:
            f.write(json.dumps(asdict(ev)) + "\n")

    # Summary
    from collections import Counter
    counts = Counter(e.task for e in trace)
    print(f"\nwrote {len(trace)} events to {out}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
