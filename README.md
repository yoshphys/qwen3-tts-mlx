# qwen3-tts-mlx

> **This repository is archived.**
> Development has moved to [yoshphys/mlx-tts](https://github.com/yoshphys/mlx-tts), which provides a more general TTS interface for Apple Silicon.

Python scripts for running [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts-6841b85e0ef8f47fa3e0a5a0) on Apple Silicon via [mlx-audio](https://github.com/Blaizzy/mlx-audio).

Two entry points are provided:

| Script | Purpose |
|--------|---------|
| `qwen3_tts_mlx.py` | CLI — single text or file |
| `run_tts.py` | TOML-driven batch runner |

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone <this-repo>
cd qwen3-tts-mlx
uv sync          # installs mlx-audio from git + soundfile
```

Or with pip:

```bash
pip install git+https://github.com/Blaizzy/mlx-audio.git soundfile
```

## Models

Three model variants are available on [mlx-community](https://huggingface.co/mlx-community):

| Variant | Method | Description |
|---------|--------|-------------|
| `*-Base-*` | `generate()` | Preset voices or voice cloning via reference audio |
| `*-CustomVoice-*` | `generate_custom_voice()` | Preset voices with emotion/style control |
| `*-VoiceDesign-*` | `generate_voice_design()` | Design any voice from a text description |

The variant is detected automatically from the model ID — no manual selection needed.

### Available models

```
mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit
mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit
mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit
mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit
mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
```

### Speakers (Base / CustomVoice)

| Language | Speakers |
|----------|---------|
| Japanese | `Vivian`, `Serena`, `Uncle_Fu`, `Dylan` (Beijing dialect), `Eric` (Sichuan dialect) |
| English  | `Ryan`, `Aiden` |

---

## CLI — `qwen3_tts_mlx.py`

### Basic usage

```bash
# Base model, preset voice
python qwen3_tts_mlx.py --text "Hello, world." --voice Ryan

# Read text from a file
python qwen3_tts_mlx.py --file input.txt --output speech.wav

# Play immediately after generation
python qwen3_tts_mlx.py --text "Hello!" --play
```

### Voice cloning (Base)

```bash
python qwen3_tts_mlx.py \
    --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
    --text "This is a cloned voice." \
    --ref-audio my_voice.wav \
    --ref-text "This is what my voice sounds like."
```

### Emotion control (CustomVoice)

```bash
python qwen3_tts_mlx.py \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --text "I'm so excited to meet you!" \
    --voice Vivian \
    --instruct "Very happy and excited."
```

### Voice design (VoiceDesign)

```bash
python qwen3_tts_mlx.py \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \
    --text "Good morning, everyone." \
    --lang-code English \
    --instruct "A calm, professional female voice speaking slowly."
```

### Streaming

```bash
python qwen3_tts_mlx.py --text "Hello, how are you today?" \
    --stream --streaming-interval 0.5
```

### All options

```
Input:
  --text, -t TEXT          Text to synthesize
  --file, -f FILE          Plain-text file to read from

Output:
  --output, -o PATH        Output file path (default: output.wav)
  --format {wav,flac,mp3}  Audio format (default: wav)

Model:
  --model, -m MODEL        HuggingFace model ID
  --list-models            Print available models and exit

Voice:
  --voice NAME             Preset speaker name
  --instruct TEXT          Emotion/style (CustomVoice) or voice description (VoiceDesign)
  --ref-audio FILE         Reference WAV for voice cloning (Base)
  --ref-text TEXT          Transcript of --ref-audio
  --lang-code CODE         Language hint: auto / English / Japanese / ... (default: auto)

Generation:
  --speed FLOAT            Speech speed multiplier (default: 1.0)
  --temperature FLOAT      Sampling temperature (default: 0.9)
  --top-k INT              Top-k sampling (default: 50)
  --top-p FLOAT            Nucleus sampling threshold (default: 1.0)
  --repetition-penalty F   Repetition penalty (default: 1.05)

Playback / Streaming:
  --play                   Play audio after generation (requires sounddevice)
  --stream                 Enable streaming generation
  --streaming-interval F   Chunk interval in seconds (default: 2.0)
```

---

## TOML runner — `run_tts.py`

Reads a TOML config file and generates one audio file per segment (or one per text in a batch segment). Models with the same ID are loaded only once and reused across segments.

```bash
python run_tts.py example.toml
```

Output is written to a directory named after the config file:

```
example/
  example_1.wav        # segment 1 (single text)
  example_2_0.wav      # segment 2, text 0 (batch)
  example_2_1.wav      # segment 2, text 1 (batch)
  ...
```

### Config structure

```toml
[settings]
# Defaults applied to every segment. All keys are optional.
model              = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
lang_code          = "auto"       # "auto" / "English" / "Japanese" / ...
format             = "wav"        # "wav" / "flac" / "mp3"
voice              = "Vivian"     # preset speaker (single text)
instruct           = "..."        # emotion/style or voice description
speed              = 1.0
temperature        = 0.9
top_k              = 50
top_p              = 1.0
repetition_penalty = 1.05
stream             = false
streaming_interval = 2.0

# Single text → generate()
[[segments]]
text = "Hello, world."

# Batch texts → batch_generate()
[[segments]]
texts    = ["Hello.", "How are you?"]
voices   = ["Ryan", "Aiden"]       # per-text speakers
instructs = ["Calm.", "Cheerful."] # per-text emotion/style
```

Segment-level keys override `[settings]`. When both `voice` and `voices` are present, `voices` takes precedence; likewise for `instruct` / `instructs`.

### TOML key reference

| Key | Scope | Default | Description |
|-----|-------|---------|-------------|
| `model` | both | `0.6B-Base-4bit` | HuggingFace model ID |
| `lang_code` | both | `"auto"` | Language hint |
| `format` | both | `"wav"` | Output format |
| `voice` | both | — | Preset speaker (single text) |
| `voices` | segment | — | Per-text speaker list (batch) |
| `instruct` | both | — | Emotion/style or voice description (single text) |
| `instructs` | segment | — | Per-text instruct list (batch) |
| `ref_audio` | both | — | Reference WAV for voice cloning (Base only) |
| `ref_text` | both | — | Transcript of `ref_audio` |
| `speed` | both | `1.0` | Speech speed multiplier |
| `temperature` | both | `0.9` | Sampling temperature |
| `top_k` | both | `50` | Top-k sampling |
| `top_p` | both | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | both | `1.05` | Repetition penalty |
| `stream` | both | `false` | Enable streaming generation |
| `streaming_interval` | both | `2.0` | Streaming chunk interval (seconds) |
| `text` | segment | — | Single text → `generate()` |
| `texts` | segment | — | Text array → `batch_generate()` |

See `example.toml` for a working config covering all three model variants.
