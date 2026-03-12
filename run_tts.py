#!/usr/bin/env python3
"""
Wrapper script: reads a TOML config and generates TTS for each segment.

Requirements:
    Python 3.11+  (tomllib is in the standard library)
    pip install mlx-audio soundfile

Usage:
    python run_tts.py config.toml

Output:
    <config_stem>/<config_stem>_<N>.wav          # single text
    <config_stem>/<config_stem>_<N>_<idx>.wav    # batch (texts array)

TOML keys (all optional unless noted):
    [settings]                 # defaults applied to every segment
    model        = "..."       # HuggingFace model ID
    lang_code    = "auto"      # language hint ("auto", "English", "Japanese", ...)
    format       = "wav"       # output format
    sample_rate  = 24000       # ignored (taken from model output); kept for compatibility
    voice        = "Vivian"    # preset voice (Base / CustomVoice)
    instruct     = "..."       # emotion/style or voice description
    speed        = 1.0
    temperature  = 0.9
    top_k        = 50
    top_p        = 1.0
    repetition_penalty = 1.05
    stream       = false
    streaming_interval = 2.0

    [[segments]]
    texts   = ["...", "..."]   # array -> batch_generate  (one file per text)
    # text  = "..."            # single string -> generate (backward compat)
    voices  = ["Vivian", "Ryan"]  # per-text voices (must match len(texts))
    # voice = "Vivian"         # same voice for all texts (expanded to list)
    # instructs = ["...", "..."]  # per-text instruct overrides
    # instruct  = "..."        # same instruct for all texts
    # ... any [settings] key can be overridden here
"""

import sys
import time
import tomllib
from pathlib import Path


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    if not config.get("segments"):
        print("[error] Config must have at least one [[segments]] entry.", file=sys.stderr)
        sys.exit(1)
    return config


def resolve(segment: dict, settings: dict, key: str, default=None):
    """Segment-level value > settings > default."""
    return segment.get(key, settings.get(key, default))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.toml>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1]).resolve()
    config      = load_config(config_path)
    settings    = config.get("settings", {})
    segments    = config["segments"]

    out_dir = config_path.parent / config_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wrap] Config  : {config_path}")
    print(f"[wrap] Output  : {out_dir}")
    print(f"[wrap] Segments: {len(segments)}")

    try:
        from mlx_audio.tts.utils import load_model
    except ImportError:
        print("[error] mlx-audio not found.\n  Run: pip install mlx-audio", file=sys.stderr)
        sys.exit(1)

    from qwen3_tts_mlx import batch_generate, generate

    model_cache: dict[str, object] = {}
    errors = []

    for seg_id, seg in enumerate(segments, start=1):
        # Resolve texts: `texts` (array) or `text` (single string)
        texts_raw = seg.get("texts")
        text_raw  = seg.get("text", "").strip()

        if texts_raw is not None:
            texts = [t.strip() for t in texts_raw if t.strip()]
        elif text_raw:
            texts = [text_raw]
        else:
            print(f"[warn] Segment {seg_id}: no texts, skipping.")
            continue

        if not texts:
            print(f"[warn] Segment {seg_id}: texts array is empty, skipping.")
            continue

        print(f"\n[wrap] --- Segment {seg_id} ({len(texts)} text(s)) ---")

        # Resolve common parameters
        model_id   = resolve(seg, settings, "model",       "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")
        lang_code  = resolve(seg, settings, "lang_code",   "auto")
        fmt        = resolve(seg, settings, "format",      "wav")
        instruct   = resolve(seg, settings, "instruct",    None)
        speed      = resolve(seg, settings, "speed",       1.0)
        temperature= resolve(seg, settings, "temperature", 0.9)
        top_k      = resolve(seg, settings, "top_k",       50)
        top_p      = resolve(seg, settings, "top_p",       1.0)
        rep_pen    = resolve(seg, settings, "repetition_penalty", 1.05)
        stream     = resolve(seg, settings, "stream",      False)
        s_interval = resolve(seg, settings, "streaming_interval", 2.0)
        ref_audio  = resolve(seg, settings, "ref_audio",   None)
        ref_text   = resolve(seg, settings, "ref_text",    None)

        # Load (or reuse) model
        if model_id not in model_cache:
            print(f"[model] Loading {model_id} ...")
            t0 = time.time()
            model_cache[model_id] = load_model(model_id)
            print(f"[model] Loaded ({time.time() - t0:.1f}s)")
        model = model_cache[model_id]

        try:
            if len(texts) == 1:
                # Single text -> generate()
                voice = resolve(seg, settings, "voice", None)
                output_path = str(out_dir / f"{config_path.stem}_{seg_id}.{fmt}")
                generate(
                    text=texts[0],
                    model=model,
                    output_path=output_path,
                    voice=voice,
                    instruct=instruct,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    lang_code=lang_code,
                    speed=speed,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                    audio_format=fmt,
                    stream=stream,
                    streaming_interval=s_interval,
                )
            else:
                # Multiple texts -> batch_generate()
                #   voices: per-text list > single voice expanded > None
                #   instructs: per-text list > single instruct expanded > None
                voices_raw   = resolve(seg, settings, "voices",   None)
                voice_raw    = resolve(seg, settings, "voice",    None)
                instructs_raw = resolve(seg, settings, "instructs", None)

                if voices_raw is not None:
                    voices = voices_raw
                elif voice_raw is not None:
                    voices = [voice_raw] * len(texts)
                else:
                    voices = None

                if instructs_raw is not None:
                    instructs = instructs_raw
                elif instruct is not None:
                    instructs = [instruct] * len(texts)
                else:
                    instructs = None

                output_prefix = str(out_dir / f"{config_path.stem}_{seg_id}")
                batch_generate(
                    texts=texts,
                    model=model,
                    output_prefix=output_prefix,
                    voices=voices,
                    instructs=instructs,
                    lang_code=lang_code,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                    audio_format=fmt,
                    stream=stream,
                    streaming_interval=s_interval,
                )
        except Exception as e:
            print(f"[error] Segment {seg_id} failed: {e}", file=sys.stderr)
            errors.append(seg_id)

    ok = len(segments) - len(errors)
    print(f"\n[wrap] Done. {ok}/{len(segments)} segments succeeded.")
    if errors:
        print(f"[wrap] Failed segments: {errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
