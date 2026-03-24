import argparse
import os
import shutil
import time
from pathlib import Path

# Critical: Set these BEFORE importing transformers/ace-step
# Prevents memory issues during model initialization on Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Mac memory management
os.environ["ACCELERATE_DISABLE_RICH"] = "1"  # Disable rich progress bars
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Reduce warnings

import torch
from acestep.handler import AceStepHandler


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 45.3s", "1h 5m 23.1s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate music with ACE-Step 1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic instrumental
  python main.py "upbeat electronic dance music" --duration 15

  # With lyrics for vocal music
  python main.py "pop ballad" --lyrics-file my_lyrics.txt --duration 20

  # With musical parameters
  python main.py "jazz piano" --bpm 120 --key-scale "C major" --time-signature "4/4"

  # Generate more variations
  python main.py "ambient soundscape" --batch-size 4 --duration 10
        """,
    )

    # Required
    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the music to generate",
    )

    # Basic options
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_music.wav",
        help="Output filename (default: generated_music.wav)",
    )

    # Quality options
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Inference steps - higher = better quality (4-32, default: 8)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="Guidance scale - higher = follows prompt more (1-15, default: 7.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (-1 for random, default: -1)",
    )

    # Musical parameters
    parser.add_argument(
        "--lyrics-file",
        type=str,
        default=None,
        help=(
            "Path to text file containing lyrics - REQUIRED for vocal music "
            "(without this, only instrumental music is generated)"
        ),
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=None,
        help="Beats per minute (e.g., 120)",
    )
    parser.add_argument(
        "--key-scale",
        type=str,
        default="",
        help="Musical key (e.g., 'C major', 'A minor', 'D#')",
    )
    parser.add_argument(
        "--time-signature",
        type=str,
        default="",
        help="Time signature (e.g., '4/4', '3/4', '6/8')",
    )
    parser.add_argument(
        "--vocal-language",
        type=str,
        default="en",
        help="Language for vocals: en, zh, ja, etc. (default: en)",
    )

    # Advanced options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of variations to generate (1-4, default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="acestep-v15-turbo",
        help=(
            "DiT model variant to use (default: acestep-v15-turbo). "
            "If you pass a 5Hz LM model name like 'acestep-5Hz-lm-1.7B', it will be treated as --lm-model."
        ),
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        default="",
        help=(
            "Optional 5Hz LM model directory name (e.g., 'acestep-5Hz-lm-1.7B'). "
            "This script does not require the LM for basic text2music, but we will ensure it's downloaded if set."
        ),
    )

    return parser.parse_args()


def load_lyrics(lyrics_file: str | None) -> str:
    """Load lyrics from file if provided.

    Args:
        lyrics_file: Path to lyrics file, or None

    Returns:
        Lyrics content as string, or empty string if no file provided

    Raises:
        SystemExit: If lyrics file is not found or cannot be read
    """
    if not lyrics_file:
        return ""

    MAX_LYRICS_CHARS = 4096

    try:
        with open(lyrics_file, encoding="utf-8") as f:
            lyrics = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Lyrics file not found: {lyrics_file}")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ Error reading lyrics file: {e}")
        raise SystemExit(1)

    if len(lyrics) > MAX_LYRICS_CHARS:
        print(f"❌ Lyrics too long: {len(lyrics):,} characters (maximum is {MAX_LYRICS_CHARS:,}).")
        print("   Please shorten your lyrics file and try again.")
        raise SystemExit(1)

    print(f"📄 Loaded lyrics from: {lyrics_file} ({len(lyrics):,} chars)")
    return lyrics


def detect_device() -> str:
    """Detect and configure the best available compute device.

    Returns:
        Device string: "mps" for Apple Silicon or "cpu" for fallback
    """
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"🍎 Using device: {device} (Apple Silicon)")
    else:
        device = "cpu"
        print(f"💻 Using device: {device} (No GPU acceleration)")
    return device


def get_project_root() -> str:
    """Get the project root path for ACE-Step models.

    Returns:
        Project root directory path where checkpoints are stored.
        Respects the ACESTEP_PROJECT_ROOT environment variable (same as the
        handler's own resolution logic), falling back to the current working
        directory so that ``./checkpoints/`` is always the default location.
    """
    env_root = os.environ.get("ACESTEP_PROJECT_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    return os.getcwd()


def is_lm_model_name(model_name: str) -> bool:
    """Return True if *model_name* looks like an ACE-Step 5Hz LM checkpoint directory."""
    return (model_name or "").strip().startswith("acestep-5Hz-lm-")


def repair_incomplete_dit_checkpoint(model_dir: Path) -> None:
    """Move aside incomplete DiT checkpoints so ACE-Step auto-download can succeed.

    ACE-Step's download precheck only tests for directory existence. If the folder exists
    but is missing required artifacts (most critically silence_latent.pt), initialization
    fails without re-downloading. We detect that case and move the folder away.
    """
    try:
        if not model_dir.is_dir():
            return
        silence_latent = model_dir / "silence_latent.pt"
        config_json = model_dir / "config.json"

        # Minimal integrity check for a DiT checkpoint folder.
        if silence_latent.is_file() and config_json.is_file():
            return

        # If either required file is missing, treat as incomplete.
        if not silence_latent.is_file() or not config_json.is_file():
            suffix = time.strftime("%Y%m%d-%H%M%S")
            backup_dir = model_dir.with_name(f"{model_dir.name}.incomplete-{suffix}")
            print("⚠️  Detected an incomplete DiT checkpoint folder:")
            print(f"   {model_dir}")
            if not silence_latent.is_file():
                print(f"   Missing: {silence_latent.name}")
            if not config_json.is_file():
                print(f"   Missing: {config_json.name}")
            print("   Moving it aside so ACE-Step can re-download a clean copy...")
            shutil.move(str(model_dir), str(backup_dir))
            print(f"   Moved to: {backup_dir}")
            print()
    except Exception as exc:
        # Non-fatal; initialization will still attempt to proceed.
        print(f"⚠️  Could not repair checkpoint folder {model_dir}: {exc}")


def initialize_model(
    model_name: str, device: str, project_root: str
) -> tuple[AceStepHandler, float, bool]:
    """Initialize the ACE-Step model handler.

    Args:
        model_name: Name of the model to load (e.g., "acestep-v15-turbo")
        device: Compute device to use ("mps" or "cpu")
        project_root: Root directory where model checkpoints are stored

    Returns:
        Tuple of (handler, load_time, success):
            - handler: Initialized AceStepHandler instance
            - load_time: Time taken to load the model in seconds
            - success: Whether initialization was successful
    """
    print("✨ Loading ACE-Step handler...")
    handler = AceStepHandler()

    print(f"📦 Loading DiT model: {model_name}")
    print("   (Model files detected in cache)")
    print()

    # Check if model exists, inform user about download if needed
    model_path = Path(project_root) / "checkpoints" / model_name
    if model_path.exists():
        print(f"✅ Model found at: {model_path}")
    else:
        print(f"📥 Model will be downloaded to: {model_path}")
        print("   (First run only - ~3-5GB download)")
        print()

    # If the folder exists but is missing required artifacts, move it aside so
    # ACE-Step's auto-download can fetch a complete checkpoint.
    repair_incomplete_dit_checkpoint(model_path)

    # Time model loading
    model_load_start = time.time()
    status, success = handler.initialize_service(
        project_root=project_root,
        config_path=model_name,
        device=device,
        use_flash_attention=False,  # Not available on MPS
        compile_model=False,  # Keep it simple for compatibility
        offload_to_cpu=False,  # Assume sufficient memory
        offload_dit_to_cpu=False,
        quantization=None,  # No quantization
        use_mlx_dit=False,  # Use PyTorch backend (MLX VAE is auto-enabled on MPS)
    )
    model_load_time = time.time() - model_load_start

    if not success:
        print(f"❌ Failed to initialize service: {status}")

    return handler, model_load_time, success


def print_generation_info(args: argparse.Namespace, lyrics: str) -> None:
    """Print information about the music generation parameters.

    Args:
        args: Parsed command-line arguments
        lyrics: Lyrics content to be used in generation
    """
    print(f"🎵 Generating: {args.prompt}")

    if lyrics:
        # Show first line or first 60 chars of lyrics
        preview = lyrics.split("\n")[0][:60]
        print(f"🎤 Lyrics: {preview}{'...' if len(lyrics) > 60 else ''}")

    # Build info string
    info_parts = [
        f"Duration: {args.duration}s",
        f"Steps: {args.steps}",
        f"Guidance: {args.guidance}",
    ]
    if args.bpm:
        info_parts.append(f"BPM: {args.bpm}")
    if args.key_scale:
        info_parts.append(f"Key: {args.key_scale}")
    if args.time_signature:
        info_parts.append(f"Time: {args.time_signature}")
    if args.batch_size != 2:
        info_parts.append(f"Batch: {args.batch_size}")

    print(f"⏱️  {' | '.join(info_parts)}")
    print()


def generate_music(
    handler: AceStepHandler, args: argparse.Namespace, lyrics: str
) -> tuple[dict, float]:
    """Generate music using the ACE-Step model.

    Args:
        handler: Initialized AceStepHandler instance
        args: Parsed command-line arguments with generation parameters
        lyrics: Lyrics content for vocal music generation

    Returns:
        Tuple of (result, generation_time):
            - result: Generation result dictionary containing audio outputs
            - generation_time: Time taken for generation in seconds
    """
    generation_start = time.time()
    result = handler.generate_music(
        captions=args.prompt,
        lyrics=lyrics,
        bpm=args.bpm,
        key_scale=args.key_scale,
        time_signature=args.time_signature,
        vocal_language=args.vocal_language,
        audio_duration=float(args.duration),
        inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_random_seed=(args.seed == -1),
        seed=args.seed,
        batch_size=args.batch_size,
        task_type="text2music",
    )
    generation_time = time.time() - generation_start
    return result, generation_time


def save_audio_outputs(result: dict, output_file: str) -> None:
    """Save generated audio outputs to file(s).

    Args:
        result: Generation result dictionary containing audio outputs
        output_file: Base output filename for saving audio
    """
    import soundfile as sf

    if "audios" not in result or not result["audios"]:
        print("❌ No audio generated")
        if "status_message" in result:
            print(f"   Status: {result['status_message']}")
        return

    audio_outputs = result["audios"]

    if len(audio_outputs) == 1:
        # Single output
        audio_data = audio_outputs[0]["tensor"]
        sample_rate = audio_outputs[0]["sample_rate"]

        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()

        # Ensure it's in the right shape for soundfile (samples, channels) or (samples,)
        if audio_data.ndim == 2:
            audio_data = audio_data.T  # soundfile expects (samples, channels)

        sf.write(output_file, audio_data, sample_rate)
        print(f"✅ Saved to {output_file}")
    else:
        # Multiple outputs (batch) - save each with a number
        for i, audio_output in enumerate(audio_outputs):
            audio_data = audio_output["tensor"]
            sample_rate = audio_output["sample_rate"]

            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()

            if audio_data.ndim == 2:
                audio_data = audio_data.T

            numbered_output = output_file.replace(".wav", f"_{i + 1}.wav")
            sf.write(numbered_output, audio_data, sample_rate)
            print(f"✅ Saved to {numbered_output}")


def print_timing_summary(
    model_load_time: float, generation_time: float, total_time: float, audio_duration: int
) -> None:
    """Print timing statistics for the music generation.

    Args:
        model_load_time: Time taken to load the model in seconds
        generation_time: Time taken to generate music in seconds
        total_time: Total elapsed time in seconds
        audio_duration: Duration of generated audio in seconds
    """
    print()
    print("⏱️  Timing Summary:")
    print(f"   Model loading: {format_duration(model_load_time)}")
    print(f"   Music generation: {format_duration(generation_time)}")
    print(f"   Total elapsed: {format_duration(total_time)}")
    print(f"   Generation speed: {generation_time / audio_duration:.2f}x realtime")


def main() -> None:
    """Main entry point for the music generation script."""
    # Parse command-line arguments
    args = parse_arguments()

    # Back-compat: if user passes an LM checkpoint name to --model, treat it as LM selection.
    dit_model = (args.model or "").strip()
    lm_model = (args.lm_model or "").strip()
    if is_lm_model_name(dit_model) and not lm_model:
        lm_model = dit_model
        dit_model = "acestep-v15-turbo"
        print("⚠️  '--model' was set to a 5Hz LM checkpoint; using it as '--lm-model' instead.")
        print(f"   LM model: {lm_model}")
        print(f"   DiT model: {dit_model}")
        print()

    # Turbo DiT checkpoints clamp diffusion steps to 8 internally.
    if "turbo" in dit_model and args.steps > 8:
        print(
            f"⚠️  '{dit_model}' clamps --steps to 8 internally; you requested {args.steps}, using 8."
        )
        args.steps = 8

    # Load lyrics if provided
    lyrics = load_lyrics(args.lyrics_file)

    # Start total timer
    total_start = time.time()

    # Detect compute device
    device = detect_device()

    # Get project root for model storage
    project_root = get_project_root()

    # If LM selection was provided, ensure it's downloaded (so passing it doesn't break).
    if lm_model:
        try:
            from acestep.model_downloader import ensure_lm_model

            checkpoints_root = Path(project_root) / "checkpoints"
            print(f"🧠 Ensuring 5Hz LM is available: {lm_model}")
            ok, msg = ensure_lm_model(model_name=lm_model, checkpoints_dir=checkpoints_root)
            if not ok:
                print(f"⚠️  LM download/availability check failed: {msg}")
            else:
                print(f"✅ LM ready: {msg}")
            print()
        except Exception as exc:
            print(f"⚠️  Skipping LM availability check (error: {exc})")
            print()

    # Initialize model
    handler, model_load_time, success = initialize_model(dit_model, device, project_root)
    if not success:
        raise SystemExit(1)

    print(f"⏱️  Model loaded in {format_duration(model_load_time)}")
    print()

    # Print generation parameters
    print_generation_info(args, lyrics)

    # Generate music
    result, generation_time = generate_music(handler, args, lyrics)

    # Save audio outputs
    save_audio_outputs(result, args.output)

    # Display timing summary if audio was generated
    if "audios" in result and result["audios"]:
        total_time = time.time() - total_start
        print_timing_summary(model_load_time, generation_time, total_time, args.duration)


if __name__ == "__main__":
    main()
