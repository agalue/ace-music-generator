# Music Generator - ACE-Step 1.5

AI-powered music generation using ACE-Step 1.5, optimized for Apple Silicon with MPS (Metal Performance Shaders) acceleration.

## Overview

This project uses the ACE-Step 1.5 model to generate music from text prompts. It's configured to work seamlessly with `uv` package manager and automatically uses MPS acceleration on Apple Silicon Macs.

**Features:**
- Apple Silicon Optimized — MPS acceleration for M1/M2/M3/M4 chips with MLX DiT backend
- High-Quality Music Generation — Instrumental and vocal music with lyrics
- Fast Performance — 15-second song in ~40-50 seconds on M4
- Musical Control — BPM, key, time signature, sampler, and more
- XL Models — 4B parameter DiT variants for higher quality (requires ~9 GB unified memory)
- Hardened Parameters — All inputs are validated against upstream constants before inference

## Requirements

- **Python**: 3.11 or 3.12
- **OS**: macOS (Apple Silicon M1/M2/M3/M4 recommended; Intel Macs fall back to CPU)
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **Disk Space**: ~5 GB for standard model files on first run; ~10 GB for XL models
- **Memory**: 8 GB minimum for standard models; 16 GB+ recommended; ~9 GB required for XL models

### Memory Usage by Model

| Model | VRAM / Unified Memory | Notes |
|---|---|---|
| `acestep-v15-turbo` | ~4.7 GB | Default; fast, 8 steps |
| `acestep-v15-sft` | ~4.7 GB | High quality, 50 steps |
| `acestep-v15-base` | ~4.7 GB | Full feature set, 50 steps |
| `acestep-v15-xl-turbo` | ~9 GB | 4B DiT, fast, 8 steps |
| `acestep-v15-xl-sft` | ~9 GB | 4B DiT, high quality, 50 steps |
| `acestep-v15-xl-base` | ~9 GB | 4B DiT, full features, 50 steps |

This project was tested on an M4 Mac Mini with 16 GB of unified memory.

## Installation

```bash
# 1. Install uv (if not already installed)
# Or: brew install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install all dependencies (ace-step included)
git clone https://github.com/agalue/ace-music-generator.git
cd ace-music-generator
uv sync

# 3. Verify installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
uv run python -c "from acestep.handler import AceStepHandler; print('ACE-Step: OK')" 2>/dev/null
```

**Expected output:**
```
PyTorch: 2.10.0, MPS: True
ACE-Step: OK
```

## Usage

### Quick Start — Instrumental Music

```bash
# Generate instrumental music (15 s, default settings)
uv run main.py "energetic rock guitar solo"

# Custom duration
uv run main.py "calm piano melody" --duration 30

# Higher quality (more steps; slower)
uv run main.py "jazz saxophone improvisation" --steps 16 --guidance 8.0
```

### Featured Example — Early-2000s Nu-Metal / Rap-Rock (Original)

Generate a 3-minute original nu-metal / rap-rock song:

**Step 1: Understand the two text inputs**

ACE-Step treats two inputs differently, so it matters what goes where:
- **`prompt` (tags field)** — comma-separated descriptive tags work best and match the model's training data format (e.g. `female lead vocalist, raspy female vocals, nu-metal, drop-tuned guitar`). Prose sentences and ALL-CAPS directives are less reliable — the DiT text encoder has a 256-token cap on the full structured prompt, so concise tags make better use of that budget than verbose instructions.
- **`--lyrics-file` (lyrics field)** — singable text with bracket section-header tags such as `[intro]`, `[verse]`, `[chorus]`, `[bridge]`, `[outro]`. The entire lyrics string (section headers included) is passed as a single token sequence to the text encoder — there is no special regex parsing of brackets in the current codebase. The headers follow the model's training data conventions and help the model segment the song structure; style hints inside the brackets (e.g. `[verse - rap]`, `[chorus - rock]`) may guide vocal delivery since the model was trained on varied bracket formats. Avoid voice prefixes like `F:` / `M:` — they are not part of any training convention and behave as literal lyric text.
- **Conservative length guard** — `main.py` rejects lyrics files exceeding 4,096 characters. This is a conservative quality proxy: the text encoder's hard truncation is at 2,048 *tokens* (roughly 8,000+ English characters), but keeping lyrics well under that budget improves lyric-to-audio alignment.

**Step 2: Generate the song:**
```bash
uv run main.py \
  "nu-metal, rap-rock, early 2000s rock, dual vocalists, male rapper, male hip-hop flow, male rhythmic rap, rap verses, female rock vocalist, raspy female vocals, female rock screaming, female belted chorus, vocal contrast, heavy distorted guitar, drop-tuned guitar, palm mute riffs, heavy bass guitar, punchy kick snare, energetic, aggressive, intense, powerful" \
  --lyrics-file sample_lyrics.txt \
  --duration 180 \
  --bpm 125 \
  --key-scale "E minor" \
  --vocal-language en \
  --output nu_metal_dual_vocals.wav \
  --batch-size 1 \
  --steps 8 \
  --guidance 7 \
  --time-signature "4/4" \
  --lm-model acestep-5Hz-lm-1.7B
```

### XL Model Example (requires ~9 GB unified memory)

```bash
# XL turbo — same speed as standard turbo but higher quality
uv run main.py "epic orchestral battle music with brass and strings" \
  --model acestep-v15-xl-turbo \
  --duration 30

# XL SFT — highest quality, 50 steps (auto-detected from model name)
uv run main.py "cinematic sci-fi atmosphere with ethereal pads" \
  --model acestep-v15-xl-sft \
  --duration 30 \
  --guidance 8.0
```

The script automatically sets `--steps` to 50 when a non-turbo model is selected.

### Output Files

**Important**: The system generates **one WAV per batch item**. By default `--batch-size` is 2, so you get two variations per prompt:
- `{stem}_1{ext}` — First variation
- `{stem}_2{ext}` — Second variation

For example, running with `--output song.wav` will create `song_1.wav` and `song_2.wav`.

### Command-Line Arguments

**Required:**
- `prompt` — Text description of the music to generate (use comma-separated tags for best results)

**Basic Options:**
- `--duration` — Length in seconds **[10–600]** (default: 15)
- `--output` — Output filename (default: `generated_music.wav`)

**Quality Options:**
- `--steps` — Inference steps **[1–100]**, higher = better quality but slower (default: auto — 8 for turbo, 50 for base/sft)
- `--guidance` — Guidance scale **[1.0–20.0]**, higher = follows prompt more closely (default: 7.0)
- `--seed` — Random seed for reproducibility (-1 for random, default: -1)

**Musical Parameters:**
- `--lyrics-file` — Path to text file containing lyrics — **REQUIRED for vocal music** (without this, only instrumental music is generated)
- `--bpm` — Beats per minute **[30–300]** (e.g., 120)
- `--key-scale` — Musical key — must be a valid key+mode string (e.g., `"C major"`, `"A minor"`, `"F# minor"`). Leave empty to let the model decide.
- `--time-signature` — Time signature with numerator in `[2, 3, 4, 6]` (e.g., `"4/4"`, `"3/4"`, `"6/8"`). Leave empty to let the model decide.
- `--vocal-language` — ISO language code for vocals (default: `en`). Invalid codes are rejected. Common values: `en`, `zh`, `ja`, `es`, `fr`, `de`, `ko`, `pt`, `ru`

**Advanced Options:**
- `--batch-size` — Number of variations to generate **[1–8]** (default: 2)
- `--model` — DiT model variant (default: `acestep-v15-turbo`). See **Model Zoo** table above.
- `--lm-model` — Optional 5Hz LM model name (e.g., `acestep-5Hz-lm-1.7B`). Ensures the model is downloaded; the LM is not required for basic text-to-music generation.
- `--infer-method` — Diffusion inference method: `ode` (deterministic, default) or `sde` (stochastic). Try `sde` for more variation between seeds.
- `--sampler` — Diffusion sampler: `euler` (fast, default) or `heun` (higher-order, slightly slower, can improve quality).
- `--download-source` — Preferred model download source: `huggingface` (default via `auto`), `modelscope`, or `auto`. Use `modelscope` if HuggingFace is slow or blocked in your region.

### First Run

On the first run, the script will automatically download the ACE-Step model files from HuggingFace. This may take several minutes depending on your internet connection. Subsequent runs use the cached models.

| Model type | Approx. download size |
|---|---|
| Standard (turbo / sft / base) | ~3–5 GB |
| XL (xl-turbo / xl-sft / xl-base) | ~8–10 GB |

### Generation Time

Generation time varies based on your hardware, the duration of audio requested, and the number of inference steps. The script displays timing information after each generation.

Example timing output:
```
Timing Summary:
   Model loading:    34.0s
   Music generation: 1m 57.8s
   Total elapsed:    2m 32.0s
   Generation speed: 0.65x realtime
```

Generation uses your Apple Silicon GPU (MPS) with MLX DiT acceleration. Time scales roughly linearly with audio duration and step count.

### Generating Vocal Music

**Important**: To generate music with vocals, you **MUST** provide a lyrics file using `--lyrics-file`. Simply describing "vocals" in the prompt will NOT produce singing — the model requires actual lyrics text.

Steps:
1. Create a text file with your lyrics (one file can contain full song lyrics)
2. Use `--lyrics-file` to point to your lyrics
3. Describe the vocal style and music genre in your prompt
4. Optionally add musical parameters like `--bpm`, `--key-scale`, etc.

Without `--lyrics-file`, you'll get instrumental music regardless of your prompt.

### Tips for Better Quality

**For Higher Quality:**
- Use more `--steps` (16–20 for important songs, up to 32 for turbo models or 50+ for base/sft)
- Try `--sampler heun` for subtle quality improvements at the cost of ~20% slower generation
- Try `--infer-method sde` for more natural variation between seeds

**Genre Considerations:**
- Electronic, EDM, and ambient music typically sound most realistic
- Rock, metal, and orchestral can sometimes sound MIDI-like
- Vocals with a lyrics file produce much better results than relying on the prompt alone

**Prompt Engineering:**
- Be specific: "live recording feel" or "realistic instruments"
- Describe the energy and mood clearly
- For rock: add instrument tags like `distorted guitar, drop-tuned guitar, palm mute riffs, heavy bass guitar, punchy kick snare`
- For vocals: add tags like `male vocalist, raspy female vocals, belted chorus, vocal fry`
- **Dual vocals with contrast**: list both roles as separate tags — e.g. `male rapper, male hip-hop flow, female rock vocalist, raspy female vocals, vocal contrast`
- **Section-level vocal hints**: add style descriptors inside bracket section headers in your lyrics file — e.g. `[verse - rap]` and `[chorus - rock]`
- **Heavy guitar**: add tags like `heavy distorted guitar, drop-tuned guitar, high-gain amp`

**Lyrics Formatting for Dual Vocalists:**
- Use concise section markers that specify vocalist + delivery: `[Verse 1 - spoken word - male]`, `[Chorus - raspy belting + scream accents - female]`
- Mark instrumentals clearly: `[Instrumental - guitar riff]`, `[Guitar Solo - instrumental]`
- Use parentheses for ad-libs/backing lines (e.g. `(yeah)`, `(turn it down)`)

**Batch Generation:**
- Use `--batch-size 2` (default) to generate variations and pick the best one
- Quality can vary noticeably between seeds even with the same prompt

### Common Warnings (Safe to Ignore)

You may see these warnings during execution — they are normal and do not affect functionality:

1. **"bitsandbytes not installed. Using standard AdamW"**
   — Expected. The standard optimizer works fine for inference.

2. **"MLX VAE decode failed... falling back to PyTorch VAE"**
   — This happens when MLX runs out of memory for very long audio. The automatic fallback works correctly.

## Technical Details

### Automatic Device Detection

The project automatically detects and uses the best available hardware acceleration:

1. **MPS** — Apple Silicon (M1/M2/M3/M4) — highest priority
2. **CPU** — Fallback for Intel Macs or when MPS is unavailable (significantly slower)

**MPS-specific optimizations applied automatically:**
- MLX DiT backend enabled for Apple Silicon (new in ACE-Step 1.5)
- MLX VAE acceleration for decode
- MPS memory watermark disabled (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`) for unified memory

### ACE-Step Handler Architecture

This project uses the ACE-Step handler-based API:

```python
import torch
from acestep.handler import AceStepHandler

device = "mps" if torch.backends.mps.is_available() else "cpu"

handler = AceStepHandler()
handler.initialize_service(
    project_root=cache_dir,
    config_path="acestep-v15-turbo",
    device=device,
    use_mlx_dit=(device == "mps"),   # MLX DiT backend on Apple Silicon
)

result = handler.generate_music(
    captions="your text prompt",
    audio_duration=15.0,
    inference_steps=8,
    guidance_scale=7.0,
    infer_method="ode",
    sampler_mode="euler",
)
```

### PyTorch Configuration

- Managed by `uv`; `ace-step` installed directly from git
- PyTorch 2.9+ from PyPI with MPS support
- Dependency overrides ensure macOS-compatible PyTorch is used (ace-step's CUDA builds are automatically overridden)

**Verify acceleration:**
```bash
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Example Prompts

```bash
# Electronic/Synth
uv run main.py "uplifting trance with ethereal pads and driving bassline" --duration 20

# Classical
uv run main.py "romantic piano piece in the style of Chopin" --duration 30

# Ambient
uv run main.py "atmospheric soundscape with warm drones" --duration 25

# Jazz
uv run main.py "smooth jazz quartet with saxophone lead" --duration 20

# Cinematic
uv run main.py "epic orchestral battle music with brass and strings" --duration 25

# Using Musical Parameters
uv run main.py "upbeat funk groove" \
  --bpm 110 --key-scale "C major" --time-signature "4/4" --duration 30

# High Quality (Heun sampler + more steps)
uv run main.py "cinematic sci-fi atmosphere" \
  --steps 20 --sampler heun --guidance 8.5 --batch-size 2 --duration 20

# XL Model (best quality, requires ~9 GB)
uv run main.py "lush neo-soul with electric piano and mellow vocals" \
  --model acestep-v15-xl-sft --duration 30
```

## Credits

- **ACE-Step**: Advanced music generation model by the ACE-Step team
- **PyTorch**: Deep learning framework with MPS support
- **uv**: Fast Python package manager by Astral

## License

See the ACE-Step repository for model licensing information.
