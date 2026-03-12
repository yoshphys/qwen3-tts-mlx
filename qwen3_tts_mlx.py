#!/usr/bin/env python3
"""
Qwen3-TTS MLX: Text-to-speech script for Apple Silicon (MLX)

Installation:
    pip install mlx-audio soundfile

Available models (mlx-community):
    Base (preset voices / voice cloning):
        mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit
        mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
        mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit
        mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit

    CustomVoice (preset voices + emotion/style control):
        mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
        mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16

    VoiceDesign (create any voice from a text description):
        mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
        mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit

Supported speakers (Base / CustomVoice):
    Chinese : Vivian, Serena, Uncle_Fu, Dylan (Beijing), Eric (Sichuan)
    English : Ryan, Aiden

Usage:
    # Base model, preset voice
    python qwen3_tts_mlx.py --text "Hello, world." --voice Ryan

    # Base model, voice cloning
    python qwen3_tts_mlx.py \\
        --text "Hello, this is a cloned voice." \\
        --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \\
        --ref-audio my_voice.wav \\
        --ref-text "This is what my voice sounds like."

    # CustomVoice model (emotion control)
    python qwen3_tts_mlx.py \\
        --text "I'm so excited to meet you!" \\
        --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \\
        --voice Vivian \\
        --lang-code English \\
        --instruct "Very happy and excited."

    # VoiceDesign model (create any voice)
    python qwen3_tts_mlx.py \\
        --text "Good morning, everyone." \\
        --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \\
        --instruct "A calm, professional female voice speaking slowly."

    # Streaming generation
    python qwen3_tts_mlx.py --text "Hello, how are you today?" --stream
"""

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    ("0.6B", "base",         "4bit"): "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    ("0.6B", "base",         "6bit"): "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit",
    ("0.6B", "base",         "bf16"): "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    ("0.6B", "customvoice",  "8bit"): "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    ("1.7B", "base",         "4bit"): "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
    ("1.7B", "base",         "8bit"): "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    ("1.7B", "customvoice",  "bf16"): "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    ("1.7B", "voicedesign",  "6bit"): "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
    ("1.7B", "voicedesign",  "bf16"): "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _save_audio(audio_mx, sample_rate: int, output_path: str, audio_format: str) -> None:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("[error] soundfile not found.\n  Run: pip install soundfile", file=sys.stderr)
        sys.exit(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.array(audio_mx), sample_rate, format=audio_format.upper())


def _play_audio(audio_mx, sample_rate: int) -> None:
    try:
        import numpy as np
        import sounddevice as sd
        sd.play(np.array(audio_mx), sample_rate)
        sd.wait()
    except ImportError:
        print("[warn] sounddevice not found; skipping playback. Run: pip install sounddevice",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Single-text generation
# ---------------------------------------------------------------------------
def generate(
    text: str,
    model,
    output_path: str,
    voice: str | None = None,
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    lang_code: str = "auto",
    speed: float = 1.0,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    audio_format: str = "wav",
    play: bool = False,
    stream: bool = False,
    streaming_interval: float = 2.0,
) -> str:
    """Generate audio for a single text via model.generate() and save to output_path.

    model.generate() routes internally to generate_custom_voice() or
    generate_voice_design() based on the loaded model type.
    Returns the output path.
    """
    import mlx.core as mx

    print(f"[tts] {text[:80]}{'...' if len(text) > 80 else ''}")
    t0 = time.time()

    kwargs: dict = dict(
        text=text,
        lang_code=lang_code,
        speed=speed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stream=stream,
        streaming_interval=streaming_interval,
    )
    if voice:     kwargs["voice"] = voice
    if instruct:  kwargs["instruct"] = instruct
    if ref_audio: kwargs["ref_audio"] = ref_audio
    if ref_text:  kwargs["ref_text"] = ref_text

    audio_chunks = []
    sample_rate = 24000  # fallback; overwritten from first result
    for result in model.generate(**kwargs):
        audio_chunks.append(result.audio)
        sample_rate = result.sample_rate

    if not audio_chunks:
        print("[error] No audio was generated.", file=sys.stderr)
        sys.exit(1)

    audio = mx.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    _save_audio(audio, sample_rate, output_path, audio_format)

    print(f"[tts] Done ({time.time() - t0:.1f}s) -> {output_path}")

    if play:
        _play_audio(audio, sample_rate)

    return output_path


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
def batch_generate(
    texts: list[str],
    model,
    output_prefix: str,
    voices: list[str] | None = None,
    instructs: list[str] | None = None,
    lang_code: str = "auto",
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    audio_format: str = "wav",
    stream: bool = False,
    streaming_interval: float = 2.0,
) -> list[str]:
    """Generate audio for multiple texts in one batched forward pass.

    Outputs: <output_prefix>_0.<fmt>, <output_prefix>_1.<fmt>, ...
    Returns a list of saved file paths ordered by text index.
    """
    import mlx.core as mx
    from collections import defaultdict

    print(f"[tts] Batch: {len(texts)} texts")
    t0 = time.time()

    kwargs: dict = dict(
        texts=texts,
        lang_code=lang_code,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stream=stream,
        streaming_interval=streaming_interval,
    )
    if voices:   kwargs["voices"] = voices
    if instructs: kwargs["instructs"] = instructs

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    chunks: dict[int, list] = defaultdict(list)
    sample_rates: dict[int, int] = {}
    output_paths: list[str | None] = [None] * len(texts)

    for result in model.batch_generate(**kwargs):
        idx = result.sequence_idx
        chunks[idx].append(result.audio)
        sample_rates[idx] = result.sample_rate
        if not stream or result.is_final_chunk:
            audio = mx.concatenate(chunks[idx]) if len(chunks[idx]) > 1 else chunks[idx][0]
            path = f"{output_prefix}_{idx}.{audio_format}"
            _save_audio(audio, sample_rates[idx], path, audio_format)
            output_paths[idx] = path
            print(f"[tts] [{idx}] saved -> {path}")

    print(f"[tts] Batch done ({time.time() - t0:.1f}s)")
    return [p for p in output_paths if p is not None]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-TTS MLX: Text-to-speech for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", "-t", help="Text to convert to speech")
    src.add_argument("--file", "-f", help="Path to a plain-text input file")

    # --- Output ---
    p.add_argument("--output", "-o", default="output.wav",
                   help="Output file path (default: output.wav)")
    p.add_argument("--format", dest="audio_format", default="wav",
                   choices=["wav", "mp3", "flac"],
                   help="Output audio format (default: wav)")

    # --- Model ---
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   help=f"mlx-community model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--list-models", action="store_true",
                   help="Print available models and exit")

    # --- Voice ---
    p.add_argument("--voice", default=None,
                   help='Preset voice name (e.g. "Ryan", "Vivian")')
    p.add_argument("--instruct", default=None,
                   help='Emotion/style (CustomVoice) or voice description (VoiceDesign)')
    p.add_argument("--ref-audio", default=None,
                   help="Reference WAV for voice cloning (Base models)")
    p.add_argument("--ref-text", default=None,
                   help="Transcript of --ref-audio")
    p.add_argument("--lang-code", default="auto",
                   help='Language code, e.g. "auto", "English", "Japanese" (default: auto)')

    # --- Generation parameters ---
    p.add_argument("--speed", type=float, default=1.0,
                   help="Speech speed multiplier (default: 1.0)")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Sampling temperature (default: 0.9)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-k sampling (default: 50)")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="Top-p (nucleus) sampling (default: 1.0)")
    p.add_argument("--repetition-penalty", type=float, default=1.05,
                   help="Repetition penalty (default: 1.05)")

    # --- Playback / streaming ---
    p.add_argument("--play", action="store_true",
                   help="Play audio after generation (requires sounddevice)")
    p.add_argument("--stream", action="store_true",
                   help="Enable streaming generation")
    p.add_argument("--streaming-interval", type=float, default=2.0,
                   help="Streaming chunk interval in seconds (default: 2.0)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available mlx-community Qwen3-TTS models:\n")
        for (size, variant, quant), mid in sorted(MODELS.items()):
            print(f"  {mid}")
        print(f"\nDefault: {DEFAULT_MODEL}")
        return

    # Resolve input text
    if args.text:
        text = args.text
    else:
        path = Path(args.file)
        if not path.exists():
            print(f"[error] File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print("[error] Input file is empty", file=sys.stderr)
            sys.exit(1)

    # Load model
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError:
        print("[error] mlx-audio not found.\n  Run: pip install mlx-audio", file=sys.stderr)
        sys.exit(1)

    print(f"[model] Loading {args.model} ...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"[model] Loaded ({time.time() - t0:.1f}s)")

    generate(
        text=text,
        model=model,
        output_path=args.output,
        voice=args.voice,
        instruct=args.instruct,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        lang_code=args.lang_code,
        speed=args.speed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        audio_format=args.audio_format,
        play=args.play,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
    )


if __name__ == "__main__":
    main()
