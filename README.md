# Music Generator - ACE-Step 1.5

AI-powered music generation using ACE-Step 1.5, optimized for Apple Silicon with MPS (Metal Performance Shaders) acceleration.

## Overview

This project uses the ACE-Step 1.5 model to generate music from text prompts. It's configured to work seamlessly with `uv` package manager and automatically uses MPS acceleration on Apple Silicon Macs.

**Features:**
- 🍎 **Apple Silicon Optimized** - MPS acceleration for M1/M2/M3/M4 chips
- 🎵 **High-Quality Music Generation** - Instrumental and vocal music with lyrics
- ⚡ **Fast Performance** - 15-second song in ~40-50 seconds
- 🎛️ **Musical Control** - BPM, key, time signature, and more

## Requirements

- **Python**: 3.11 or 3.12
- **OS**: macOS (Apple Silicon M1/M2/M3/M4 recommended, also works on Intel Macs with CPU)
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **Disk Space**: ~5GB for model files on first run
- **Memory**: 8GB minimum, 16GB+ recommended for longer audio

### Memory Usage

This project was tested on an M4 Mac Mini with 16GB of unified memory. Apple Silicon uses unified memory shared between CPU and GPU.

## Installation

```bash
# 1. Install uv (if not already installed)
# Or, try `brew install uv`
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

### Quick Start - Instrumental Music

```bash
# Generate instrumental music
uv run main.py "energetic rock guitar solo"

# Specify custom duration (in seconds)
uv run main.py "calm piano melody" --duration 30

# Adjust generation quality (more steps = higher quality but slower)
uv run main.py "jazz saxophone improvisation" --steps 16 --guidance 8.0
```

### Featured Example - Early-2000s Nu-Metal / Rap-Rock (Original)

Generate a 3-minute original nu-metal / rap-rock song with an early-2000s mainstream nu-metal/rap-rock arrangement (original composition; do not clone any real singer’s voice):

**Step 1: Use the provided sample lyrics**

The repository includes `sample_lyrics.txt` with original lyrics designed for nu-metal / rap-rock dynamics.

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
  --vocal-language "en" \
  --output nu_metal_dual_vocals.wav \
  --batch-size 1 \
  --steps 8 \
  --guidance 7 \
  --time-signature "4/4" \
  --lm-model acestep-5Hz-lm-1.7B
```

This will generate `nu_metal_dual_vocals.wav` - a full 3-minute original rock song featuring:
- **Male rapper (verses)**: Tight rhythmic hip-hop flow with rap delivery — drives the verse sections
- **Female rock vocalist (pre-choruses, choruses, bridge, outro)**: Raw, raspy rock singing with belted choruses and hard screams on peak hook words
- **Heavy guitar**: Drop-tuned rhythm guitars with realistic high-gain amp tone, tight palm-mutes, big low end
- **Dynamic structure**: Intro → male rap verse → female pre-chorus → female chorus → male rap verse 2 → female pre-chorus → female chorus → half-time bridge → final chorus → female outro

The script will display timing information showing how long the generation took on your hardware.

### Output Files

**Important**: The system generates **one WAV per batch item**. By default `--batch-size` is 2, so you get two variations per prompt:
- `{output}_1.wav` - First variation
- `{output}_2.wav` - Second variation

For example, running with `--output song.wav` will create `song_1.wav` and `song_2.wav`.

### Command-Line Arguments

**Required:**
- `prompt` - Text description of the music to generate

**Basic Options:**
- `--duration` - Length in seconds (default: 15)
- `--output` - Output filename (default: generated_music.wav)

**Quality Options:**
- `--steps` - Inference steps, higher = better quality (4-32, default: 8)
- `--guidance` - Guidance scale, higher = follows prompt more closely (1-15, default: 7.0)
- `--seed` - Random seed for reproducibility (-1 for random, default: -1)

**Musical Parameters:**
- `--lyrics-file` - Path to text file containing lyrics - **REQUIRED for vocal music** (without this, only instrumental music is generated)
- `--bpm` - Beats per minute (e.g., 120)
- `--key-scale` - Musical key (e.g., "C major", "A minor", "D#")
- `--time-signature` - Time signature (e.g., "4/4", "3/4", "6/8")
- `--vocal-language` - Language for vocals: en, zh, ja, etc. (default: en)

**Advanced Options:**
- `--batch-size` - Number of variations to generate (1-4, default: 2)
- `--model` - Model variant (default: acestep-v15-turbo)

### First Run

On the first run, the script will automatically download the ACE-Step model files (~3-5GB) from HuggingFace. This may take several minutes depending on your internet connection. Subsequent runs will use the cached models.

### Generation Time

Generation time varies based on your hardware, the duration of audio requested, and the number of inference steps. The script will display actual timing information after each generation, including:
- Model loading time
- Audio generation time
- Total elapsed time
- Generation speed (as multiple of realtime)

Example timing output:
```
⏱️  Timing Summary:
   Model loading: 34.0s
   Music generation: 1m 57.8s
   Total elapsed: 2m 32.0s
   Generation speed: 0.65x realtime
```

Generation uses your Apple Silicon GPU (MPS) for acceleration. Time scales roughly linearly with audio duration and step count.

### Generating Vocal Music

⚠️ **Important**: To generate music with vocals, you **MUST** provide a lyrics file using `--lyrics-file`. Simply describing "vocals" in the prompt will NOT produce singing - the model requires actual lyrics text.

See the **Featured Example** in the Usage section above for a complete nu-metal / rap-rock song with vocals. The key steps are:

1. Create a text file with your lyrics (one file can contain full song lyrics)
2. Use `--lyrics-file` to point to your lyrics
3. Describe the vocal style and music genre in your prompt
4. Optionally add musical parameters like `--bpm`, `--key-scale`, etc.

Without `--lyrics-file`, you'll only get instrumental music regardless of your prompt.

### Tips for Better Quality

AI music generation is still evolving. Here are tips to get the best results:

**For Higher Quality:**
- Use more `--steps` (16-20 for important songs, up to 32 for best quality)
- Higher steps take longer but produce more refined audio
- Default 8 steps is fast but may sound synthetic

**Genre Considerations:**
- Electronic, EDM, and ambient music typically sound more realistic
- Rock, metal, and orchestral can sometimes sound MIDI-like
- Vocals with lyrics file produce much better results than instrumental-only

**Prompt Engineering:**
- Be specific: "live recording feel" or "realistic instruments"
- Describe the energy and mood clearly
- For rock: add instrument tags like `distorted guitar, drop-tuned guitar, palm mute riffs, heavy bass guitar, punchy kick snare`
- For vocals: add tags like `male vocalist, raspy female vocals, belted chorus, vocal fry`
- **Dual vocals with contrast**: list both roles as separate tags — e.g. `male rapper, male hip-hop flow, female rock vocalist, raspy female vocals, vocal contrast`. Prose instructions like "the male should rap" are not part of the model's training distribution and are ignored.
- **Section-level vocal hints**: add style descriptors inside bracket section headers in your lyrics file — e.g. `[verse - rap]` and `[chorus - rock]` — to reinforce per-section delivery alongside the caption tags.
- **Heavy guitar**: add tags like `heavy distorted guitar, drop-tuned guitar, high-gain amp` for heavier tone
- **Complex solos**: add tags like `guitar solo, melodic lead guitar, shredding` for more sophisticated instrumental sections

**Lyrics Formatting for Dual Vocalists:**
- Use concise section markers that specify vocalist + delivery: `[Verse 1 - spoken word - male]`, `[Verse 1 - sing-rap - male]`, `[Chorus - raspy belting + scream accents - female]`
- Keep tags short so they don’t “leak” into vocals; put detailed production notes in the main prompt
- Mark instrumentals clearly: `[Instrumental - guitar riff]`, `[Guitar Solo - instrumental]`, `[Breakdown - half-time - aggressive]`
- Use parentheses for ad-libs/backing lines (e.g. `(yeah)`, `(turn it down)`)

**Batch Generation:**
- Use `--batch-size 2` (default) to generate variations
- Listen to both files - quality can vary between generations
- Keep the better-sounding version

### Common Warnings (Safe to Ignore)

You may see these warnings during execution - they are normal and don't affect functionality:

1. **"bitsandbytes not installed. Using standard AdamW"**
   - This is expected. The standard optimizer works fine for inference.

2. **"MLX VAE decode failed... falling back to PyTorch VAE"**
   - This happens when MLX runs out of memory for very long audio.
   - The automatic fallback to PyTorch works perfectly.
   - No action needed.

## Technical Details

### Automatic Device Detection

The project automatically detects and uses the best available hardware acceleration:

**Device priority:**
1. **MPS** - Apple Silicon (M1/M2/M3/M4) - highest priority
2. **CPU** - Fallback when MPS is not available (Intel Macs, or compatibility issues)

**Device-specific optimizations:**
- **MPS**: MLX VAE acceleration for Apple Silicon, memory management optimizations
- **CPU**: Standard PyTorch backend (significantly slower)

Example device detection output:
```bash
# On Apple Silicon Mac:
🍎 Using device: mps (Apple Silicon)

# On Intel Mac or CPU fallback:
💻 Using device: cpu (No GPU acceleration)
```

### Memory Management

**Environment variable** (automatically configured):
```python
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
```
This optimizes memory management on Apple Silicon unified memory architecture.

### ACE-Step Handler Architecture

This project uses the ACE-Step handler-based API with automatic device detection:

```python
import torch
from acestep.handler import AceStepHandler

# Auto-detect best device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Initialize handler
handler = AceStepHandler()

# Initialize service with model
handler.initialize_service(
    project_root=cache_dir,
    config_path="acestep-v15-turbo",
    device=device,  # auto-detected: mps or cpu
)

# Generate music
result = handler.generate_music(
    captions="your text prompt",
    audio_duration=15.0,
    inference_steps=8,
    guidance_scale=7.0,
)
```

### PyTorch Configuration

This project uses standard PyTorch from PyPI with MPS (Metal Performance Shaders) support for Apple Silicon.

**Installation:**
- Uses `uv` for dependency management
- ace-step installed directly from git as a dependency in `pyproject.toml`
- PyTorch 2.9+ with MPS support from PyPI
- Dependency overrides ensure macOS-compatible PyTorch is used (ace-step's CUDA builds are automatically overridden)

**Verify your acceleration is working:**

```bash
# Check MPS availability
uv run python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"

# Check which device will be used
uv run python -c "import torch; device = 'mps' if torch.backends.mps.is_available() else 'cpu'; print(f'Device: {device}')"
```

### Performance Characteristics

**Apple Silicon (MPS backend):**
- GPU acceleration via Metal Performance Shaders
- Unified memory architecture (no CPU↔GPU transfer overhead)
- Optimized for M-series chips (M1/M2/M3/M4)
- Excellent performance for music generation tasks

## Example Prompts

Try these example prompts for different genres and techniques:

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

# Using Musical Parameters (BPM, key, time signature)
uv run main.py "upbeat funk groove" \
  --bpm 110 --key-scale "C major" --time-signature "4/4" \
  --duration 30

# High Quality Generation (more steps and guidance)
uv run main.py "cinematic sci-fi atmosphere" \
  --steps 20 --guidance 8.5 --batch-size 2 --duration 20
```

## Credits

- **ACE-Step**: Advanced music generation model by ACE-Step team
- **PyTorch**: Deep learning framework with MPS support  
- **uv**: Fast Python package manager by Astral

## License

See the ACE-Step repository for model licensing information.
