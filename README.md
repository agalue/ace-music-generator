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

Apple Silicon uses unified memory (shared between CPU and GPU):
- **15-second songs**: ~4-6GB
- **30-second songs**: ~6-10GB
- **60-second songs**: ~10-16GB
- **120-second songs**: ~16-24GB

## Installation

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install all dependencies (ace-step included)
git clone <your-repo-url>
cd music-generator
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
uv run python main.py "energetic rock guitar solo"

# Specify custom duration (in seconds)
uv run python main.py "calm piano melody" --duration 30

# Adjust generation quality (more steps = higher quality but slower)
uv run python main.py "jazz saxophone improvisation" --steps 16 --guidance 8.0
```

### Featured Example - Linkin Park-Inspired Rock with Vocals

Generate a 3-minute original nu-metal rock song in the style of Linkin Park with contrasting male rapper and female rock vocalist:

**Step 1: Use the provided sample lyrics**

The repository includes `sample_lyrics.txt` with original lyrics inspired by Linkin Park's emotional and powerful style. The lyrics include detailed section markers specifying:
- **Vocal style instructions**: `[Verse 1 - Male Rapper: aggressive hip-hop delivery]` and `[Verse 1 - Female Lead Singer: powerful raspy rock vocals with raw emotion]`
- **Song structure**: Verse, Chorus, Pre-Chorus, Bridge, Guitar Solo
- **Intensity markers**: Help guide dynamic shifts between sections

These detailed markers help the AI model understand the intended vocal delivery and create more distinct vocal performances.

**Step 2: Generate the song:**
```bash
uv run python main.py \
  "original nu-metal rock inspired by Linkin Park style with TWO DISTINCT vocalists: first vocalist is aggressive MALE RAPPER with hip-hop delivery and rhythmic spoken word, second vocalist is powerful FEMALE ROCK SINGER with raspy intense screaming vocals and raw emotional delivery, EXTREMELY HEAVY distorted guitar riffs with aggressive tone and complex melodic shredding solo, thick bass, electronic elements and synths, pounding explosive drums with double bass, raw energy, dramatic contrast between male rap verses and female screaming chorus, alternative metal atmosphere" \
  --lyrics-file sample_lyrics.txt \
  --duration 180 \
  --bpm 125 \
  --key-scale "E minor" \
  --vocal-language "en" \
  --output linkin_park_style.wav \
  --batch-size 1 \
  --steps 16
```

This will generate `linkin_park_style.wav` - a full 3-minute original rock song featuring:
- **Two vocalists with maximum contrast**: 
  - **Male Rapper**: Aggressive hip-hop delivery with rhythmic spoken word for verses
  - **Female Rock Singer**: Powerful raspy screaming vocals with raw emotion for choruses (like current Linkin Park vocalist Emily Armstrong)
- **Heavy guitar**: Extremely distorted aggressive guitar riffs with thick, powerful tone
- **Complex guitar solo**: Intricate melodic shredding with heavy distortion
- **Electronic elements**: Synths and industrial sounds
- **Dynamic structure**: Intro, male rap verses alternating with female rock screaming, pre-chorus builds, explosive choruses, bridge with both vocalists, intricate guitar solo, outro
- **Clear vocal distinction**: Strong contrast between male hip-hop rap style and female raspy rock screaming

The script will display timing information showing how long the generation took on your hardware.

### Output Files

**Important**: The system generates **2 audio variations** per prompt (batch size = 2), giving you multiple options to choose from:
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

See the **Featured Example** in the Usage section above for a complete Linkin Park-style rock song with vocals. The key steps are:

1. Create a text file with your lyrics (one file can contain full song lyrics)
2. Use `--lyrics-file` to point to your lyrics
3. Describe the vocal style and music genre in your prompt
4. Optionally add musical parameters like `--bpm`, `--key-scale`, etc.

**Quick vocal example:**
```bash
# Create simple lyrics
echo "La la la, singing in the rain\nDancing through the pain" > simple_lyrics.txt

# Generate pop song with vocals
uv run python main.py "upbeat pop song with cheerful female vocals" \
  --lyrics-file simple_lyrics.txt \
  --duration 20
```

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
- For rock: mention specific instruments like "EXTREMELY HEAVY distorted electric guitar" and "pounding explosive drums"
- For vocals: describe voice character like "powerful male vocals" or "soft female voice"
- **Dual vocals with gender contrast**: For maximum vocal distinction (like Linkin Park), use "TWO DISTINCT vocalists" and specify genders: "first vocalist is aggressive MALE RAPPER with hip-hop delivery, second vocalist is powerful FEMALE ROCK SINGER with raspy intense screaming vocals"
- **Emphasize contrast**: Use phrases like "dramatic contrast between male rap verses and female screaming chorus" to help the model create distinctly different vocal performances
- **Heavy guitar**: Use emphatic descriptors like "EXTREMELY HEAVY distorted guitar", "aggressive tone", "thick powerful distortion" for heavier guitar sound
- **Complex solos**: Specify "intricate melodic shredding", "complex guitar solo", "technical lead guitar" for more sophisticated instrumental sections

**Lyrics Formatting for Dual Vocalists:**
- Use detailed section markers that specify WHICH vocalist and WHAT style: `[Verse 1 - Male Rapper: aggressive hip-hop delivery]` or `[Chorus - Female Lead Singer: explosive raspy screaming vocals]`
- Specify gender for maximum vocal distinction: "Male Rapper" vs "Female Lead Singer"
- Be specific about vocal characteristics: "powerful raspy rock vocals with raw emotion", "rhythmic spoken word with attitude", "intense raspy vocals with grit"
- Mark instrumental sections with detail: `[Guitar Solo - Instrumental: intricate melodic shredding with heavy distortion]`
- Specify when both vocalists perform: `[Bridge - Both Vocalists: rapper and female singer alternating and harmonizing]`
- The more detailed your vocal style descriptions and gender specifications, the better the model can create distinct vocal performances

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
uv run python main.py "uplifting trance with ethereal pads and driving bassline" --duration 20

# Classical
uv run python main.py "romantic piano piece in the style of Chopin" --duration 30

# Ambient
uv run python main.py "atmospheric soundscape with warm drones" --duration 25

# Jazz
uv run python main.py "smooth jazz quartet with saxophone lead" --duration 20

# Cinematic
uv run python main.py "epic orchestral battle music with brass and strings" --duration 25

# Using Musical Parameters (BPM, key, time signature)
uv run python main.py "upbeat funk groove" \
  --bpm 110 --key-scale "C major" --time-signature "4/4" \
  --duration 30

# High Quality Generation (more steps and guidance)
uv run python main.py "cinematic sci-fi atmosphere" \
  --steps 20 --guidance 8.5 --batch-size 2 --duration 20
```

## Credits

- **ACE-Step**: Advanced music generation model by ACE-Step team
- **PyTorch**: Deep learning framework with MPS support  
- **uv**: Fast Python package manager by Astral

## License

See the ACE-Step repository for model licensing information.
