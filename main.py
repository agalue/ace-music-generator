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
from acestep.constants import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    VALID_KEYSCALES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)
from acestep.handler import AceStepHandler

# ---------------------------------------------------------------------------
# Known DiT model names exposed to the user.  Keeping this list here (rather
# than inside parse_arguments) makes it easy to extend when upstream adds new
# checkpoints.
# ---------------------------------------------------------------------------
DIT_MODELS = [
    # Standard models (≈4.7 GB VRAM)
    "acestep-v15-turbo",
    "acestep-v15-sft",
    "acestep-v15-base",
    # XL models (≈9 GB VRAM)
    "acestep-v15-xl-turbo",
    "acestep-v15-xl-sft",
    "acestep-v15-xl-base",
]

# VRAM threshold (GB) at which we warn the user about XL model requirements.
XL_VRAM_REQUIREMENT_GB = 9

# Steps defaults: turbo models are designed for 8 steps; base/sft need ~50.
TURBO_DEFAULT_STEPS = 8
NON_TURBO_DEFAULT_STEPS = 50


# ---------------------------------------------------------------------------
# Argparse type-validator factories
# ---------------------------------------------------------------------------


def int_range(lo: int, hi: int):
    """Return an argparse *type* callable that accepts integers in [lo, hi]."""

    def _validate(value: str) -> int:
        try:
            v = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected an integer, got: {value!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"Value {v} is out of range [{lo}, {hi}]")
        return v

    _validate.__name__ = f"int[{lo},{hi}]"
    return _validate


def float_range(lo: float, hi: float):
    """Return an argparse *type* callable that accepts floats in [lo, hi]."""

    def _validate(value: str) -> float:
        try:
            v = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected a number, got: {value!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"Value {v} is out of range [{lo}, {hi}]")
        return v

    _validate.__name__ = f"float[{lo},{hi}]"
    return _validate


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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


def is_lm_model_name(model_name: str) -> bool:
    """Return True if *model_name* looks like an ACE-Step 5Hz LM checkpoint directory."""
    return (model_name or "").strip().startswith("acestep-5Hz-lm-")


def is_turbo_model(model_name: str) -> bool:
    """Return True if *model_name* refers to a turbo DiT checkpoint."""
    return "turbo" in (model_name or "").lower()


def is_xl_model(model_name: str) -> bool:
    """Return True if *model_name* refers to an XL (4B) DiT checkpoint."""
    return "-xl-" in (model_name or "").lower()


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
            print("Warning: Detected an incomplete DiT checkpoint folder:")
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
        print(f"Warning: Could not repair checkpoint folder {model_dir}: {exc}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse and lightly validate command-line arguments.

    Hard bounds for numeric parameters come directly from the upstream
    `acestep.constants` module so they stay in sync automatically.

    Returns:
        Parsed arguments namespace
    """
    model_list = "\n  ".join(DIT_MODELS)
    lm_list = (
        "  acestep-5Hz-lm-0.6B (≈1.2 GB)\n"
        "  acestep-5Hz-lm-1.7B (≈3.4 GB)\n"
        "  acestep-5Hz-lm-4B   (≈8 GB)"
    )
    valid_ts = ", ".join(str(n) for n in sorted(VALID_TIME_SIGNATURES))

    parser = argparse.ArgumentParser(
        description="Generate music with ACE-Step 1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic instrumental
  python main.py "upbeat electronic dance music" --duration 15

  # With lyrics for vocal music
  python main.py "pop ballad" --lyrics-file my_lyrics.txt --duration 20

  # With musical parameters
  python main.py "jazz piano" --bpm 120 --key-scale "C major" --time-signature "4/4"

  # Generate more variations
  python main.py "ambient soundscape" --batch-size 4 --duration 10

  # High-quality generation with an XL model (requires ~9 GB VRAM)
  python main.py "epic orchestral" --model acestep-v15-xl-sft --steps 50 --duration 30

Available DiT models:
  {model_list}

Available 5Hz LM models:
{lm_list}

Valid time-signature numerators: {valid_ts}
        """,
    )

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the music to generate",
    )

    # ------------------------------------------------------------------ #
    # Basic options                                                        #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--duration",
        type=int_range(DURATION_MIN, DURATION_MAX),
        default=15,
        help=f"Duration in seconds [{DURATION_MIN}-{DURATION_MAX}] (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_music.wav",
        help="Output filename (default: generated_music.wav)",
    )

    # ------------------------------------------------------------------ #
    # Quality options                                                      #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--steps",
        type=int_range(1, 100),
        default=None,
        help=(
            "Inference steps — higher = better quality [1-100]. "
            "Defaults to 8 for turbo models and 50 for base/sft models."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float_range(1.0, 20.0),
        default=7.0,
        help="Guidance scale — higher = follows prompt more closely [1.0-20.0] (default: 7.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (-1 for random, default: -1)",
    )

    # ------------------------------------------------------------------ #
    # Musical parameters                                                   #
    # ------------------------------------------------------------------ #
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
        type=int_range(BPM_MIN, BPM_MAX),
        default=None,
        help=f"Beats per minute [{BPM_MIN}-{BPM_MAX}] (e.g., 120)",
    )
    parser.add_argument(
        "--key-scale",
        type=str,
        default="",
        help=(
            "Musical key — must match one of the valid key+mode combinations "
            "(e.g., 'C major', 'A minor', 'F# minor'). "
            "Leave empty to let the model decide."
        ),
    )
    parser.add_argument(
        "--time-signature",
        type=str,
        default="",
        help=(
            f"Time signature — numerator must be one of [{valid_ts}] "
            "(e.g., '4/4', '3/4', '6/8'). Leave empty to let the model decide."
        ),
    )
    parser.add_argument(
        "--vocal-language",
        type=str,
        choices=VALID_LANGUAGES,
        default="en",
        metavar="LANG",
        help=(f"Language code for vocals (default: en). Valid codes: {', '.join(VALID_LANGUAGES)}"),
    )

    # ------------------------------------------------------------------ #
    # Advanced / model options                                             #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--batch-size",
        type=int_range(1, 8),
        default=2,
        help="Number of variations to generate [1-8] (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="acestep-v15-turbo",
        help=(
            "DiT model variant (default: acestep-v15-turbo). "
            "XL models (acestep-v15-xl-*) require ~9 GB VRAM. "
            "If you pass a 5Hz LM name like 'acestep-5Hz-lm-1.7B' it is "
            "automatically treated as --lm-model."
        ),
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        default="",
        help=(
            "Optional 5Hz LM model directory name (e.g., 'acestep-5Hz-lm-1.7B'). "
            "Ensures the model is downloaded; basic text2music does not require the LM."
        ),
    )
    parser.add_argument(
        "--infer-method",
        type=str,
        choices=["ode", "sde"],
        default="ode",
        help=(
            "Diffusion inference method: 'ode' (deterministic) or 'sde' (stochastic) (default: ode)"
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["euler", "heun"],
        default="euler",
        help="Diffusion sampler: 'euler' (fast) or 'heun' (higher-order, slower) (default: euler)",
    )
    parser.add_argument(
        "--download-source",
        type=str,
        choices=["huggingface", "modelscope", "auto"],
        default="auto",
        help=(
            "Preferred model download source: "
            "'huggingface', 'modelscope', or 'auto' (tries HuggingFace first) "
            "(default: auto)"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers (post-parse, where argparse choices aren't enough)
# ---------------------------------------------------------------------------


def validate_key_scale(key_scale: str) -> None:
    """Validate --key-scale against the upstream list of valid combinations.

    Args:
        key_scale: The key+mode string supplied by the user (may be empty).

    Raises:
        SystemExit: If the value is non-empty and not in VALID_KEYSCALES.
    """
    if not key_scale:
        return
    # Normalize: strip whitespace, title-case for comparison
    normalized = key_scale.strip()
    if normalized not in VALID_KEYSCALES:
        # Try case-insensitive match to give a better error message
        lower = normalized.lower()
        suggestions = [ks for ks in VALID_KEYSCALES if ks.lower() == lower]
        print(f"Error: Invalid --key-scale value: {key_scale!r}")
        if suggestions:
            print(f"   Did you mean: {suggestions[0]!r}?")
        else:
            # Show a compact sample of valid values
            sample = list(VALID_KEYSCALES)[:12]
            print(f"   Valid examples: {', '.join(repr(s) for s in sample)}, ...")
            print("   Leave --key-scale empty to let the model decide.")
        raise SystemExit(1)


def validate_time_signature(time_signature: str) -> None:
    """Validate --time-signature numerator against VALID_TIME_SIGNATURES.

    The upstream model only supports numerators in VALID_TIME_SIGNATURES
    (currently [2, 3, 4, 6]).  The denominator is informational.

    Args:
        time_signature: e.g. "4/4", "3/4", "6/8" (may be empty).

    Raises:
        SystemExit: If the numerator is not in VALID_TIME_SIGNATURES.
    """
    if not time_signature:
        return
    parts = time_signature.strip().split("/")
    try:
        numerator = int(parts[0])
    except (ValueError, IndexError):
        print(f"Error: Cannot parse --time-signature: {time_signature!r}")
        print("   Expected format: '<numerator>/<denominator>' (e.g., '4/4', '3/4', '6/8')")
        raise SystemExit(1)
    if numerator not in VALID_TIME_SIGNATURES:
        valid_ts = ", ".join(str(n) for n in sorted(VALID_TIME_SIGNATURES))
        print(
            f"Error: Invalid time-signature numerator {numerator!r} in {time_signature!r}. "
            f"Allowed numerators: {valid_ts}"
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Lyrics loading
# ---------------------------------------------------------------------------


def load_lyrics(lyrics_file: str | None) -> str:
    """Load lyrics from file if provided.

    Args:
        lyrics_file: Path to lyrics file, or None

    Returns:
        Lyrics content as string, or empty string if no file provided

    Raises:
        SystemExit: If lyrics file is not found, cannot be read, or is too long
    """
    if not lyrics_file:
        return ""

    MAX_LYRICS_CHARS = 4096

    try:
        with open(lyrics_file, encoding="utf-8") as f:
            lyrics = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Lyrics file not found: {lyrics_file}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: Could not read lyrics file: {e}")
        raise SystemExit(1)

    if len(lyrics) > MAX_LYRICS_CHARS:
        print(
            f"Error: Lyrics too long: {len(lyrics):,} characters (maximum is {MAX_LYRICS_CHARS:,})."
        )
        print("   Please shorten your lyrics file and try again.")
        raise SystemExit(1)

    print(f"Loaded lyrics from: {lyrics_file} ({len(lyrics):,} chars)")
    return lyrics


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def detect_device() -> str:
    """Detect and configure the best available compute device.

    Returns:
        Device string: "mps" for Apple Silicon or "cpu" for fallback
    """
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device} (Apple Silicon)")
    else:
        device = "cpu"
        print(f"Using device: {device} (No GPU acceleration)")
    return device


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def initialize_model(
    model_name: str,
    device: str,
    project_root: str,
    prefer_source: str = "auto",
) -> tuple[AceStepHandler, float, bool]:
    """Initialize the ACE-Step model handler.

    Args:
        model_name: Name of the model to load (e.g., "acestep-v15-turbo")
        device: Compute device to use ("mps" or "cpu")
        project_root: Root directory where model checkpoints are stored
        prefer_source: Preferred download source ("huggingface", "modelscope", "auto")

    Returns:
        Tuple of (handler, load_time, success):
            - handler: Initialized AceStepHandler instance
            - load_time: Time taken to load the model in seconds
            - success: Whether initialization was successful
    """
    print("Loading ACE-Step handler...")
    handler = AceStepHandler()

    print(f"Loading DiT model: {model_name}")

    # Check if model exists, inform user about download if needed
    model_path = Path(project_root) / "checkpoints" / model_name
    if model_path.exists():
        print(f"Model found at: {model_path}")
    else:
        print(f"Model will be downloaded to: {model_path}")
        print("   (First run only - download may take several minutes)")
    print()

    # If the folder exists but is missing required artifacts, move it aside so
    # ACE-Step's auto-download can fetch a complete checkpoint.
    repair_incomplete_dit_checkpoint(model_path)

    # Enable MLX DiT backend on Apple Silicon for better performance.
    # The upstream default is already True; we make this explicit and device-aware.
    use_mlx_dit = device == "mps"

    # Translate "auto" to None so the upstream resolver picks the best source.
    prefer_source_arg = None if prefer_source == "auto" else prefer_source

    # Time model loading
    model_load_start = time.time()
    status, success = handler.initialize_service(
        project_root=project_root,
        config_path=model_name,
        device=device,
        use_flash_attention=False,  # Not available on MPS
        compile_model=False,  # Keep it simple for compatibility
        offload_to_cpu=False,  # Assume sufficient memory on Apple Silicon unified RAM
        offload_dit_to_cpu=False,
        quantization=None,  # No quantization needed on MPS (unified memory)
        use_mlx_dit=use_mlx_dit,  # MLX DiT acceleration for Apple Silicon
        prefer_source=prefer_source_arg,
    )
    model_load_time = time.time() - model_load_start

    if not success:
        print(f"Error: Failed to initialize service: {status}")

    return handler, model_load_time, success


# ---------------------------------------------------------------------------
# Generation info display
# ---------------------------------------------------------------------------


def print_generation_info(args: argparse.Namespace, lyrics: str) -> None:
    """Print information about the music generation parameters.

    Args:
        args: Parsed command-line arguments
        lyrics: Lyrics content to be used in generation
    """
    print(f"Generating: {args.prompt}")

    if lyrics:
        # Show first line or first 60 chars of lyrics
        preview = lyrics.split("\n")[0][:60]
        print(f"Lyrics: {preview}{'...' if len(lyrics) > 60 else ''}")

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
    if args.infer_method != "ode" or args.sampler != "euler":
        info_parts.append(f"Sampler: {args.infer_method}/{args.sampler}")

    print(f"Settings: {' | '.join(info_parts)}")
    print()


# ---------------------------------------------------------------------------
# Music generation
# ---------------------------------------------------------------------------


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
        infer_method=args.infer_method,
        sampler_mode=args.sampler,
    )
    generation_time = time.time() - generation_start
    return result, generation_time


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------


def save_audio_outputs(result: dict, output_file: str) -> None:
    """Save generated audio outputs to file(s).

    For a single output the file is saved as-is.  For a batch the outputs
    are saved as ``{stem}_1{suffix}``, ``{stem}_2{suffix}``, etc., using
    :class:`pathlib.Path` manipulation so the logic is correct regardless of
    the file extension.

    Args:
        result: Generation result dictionary containing audio outputs
        output_file: Base output filename for saving audio
    """
    import soundfile as sf

    if "audios" not in result or not result["audios"]:
        print("Error: No audio generated")
        if "status_message" in result:
            print(f"   Status: {result['status_message']}")
        return

    audio_outputs = result["audios"]
    output_path = Path(output_file)

    def _write(audio_data, sample_rate: int, path: Path) -> None:
        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()
        # soundfile expects (samples, channels) for multi-channel audio
        if audio_data.ndim == 2:
            audio_data = audio_data.T
        sf.write(str(path), audio_data, sample_rate)
        print(f"Saved: {path}")

    if len(audio_outputs) == 1:
        _write(audio_outputs[0]["tensor"], audio_outputs[0]["sample_rate"], output_path)
    else:
        for i, audio_output in enumerate(audio_outputs):
            numbered = output_path.with_name(f"{output_path.stem}_{i + 1}{output_path.suffix}")
            _write(audio_output["tensor"], audio_output["sample_rate"], numbered)


# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------


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
    print("Timing Summary:")
    print(f"   Model loading:    {format_duration(model_load_time)}")
    print(f"   Music generation: {format_duration(generation_time)}")
    print(f"   Total elapsed:    {format_duration(total_time)}")
    print(f"   Generation speed: {generation_time / audio_duration:.2f}x realtime")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the music generation script."""
    # Parse command-line arguments
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Model name resolution                                                #
    # ------------------------------------------------------------------ #
    dit_model = (args.model or "").strip()
    lm_model = (args.lm_model or "").strip()

    # Back-compat: if the user passes an LM checkpoint name to --model, treat
    # it as an implicit --lm-model selection.
    if is_lm_model_name(dit_model) and not lm_model:
        lm_model = dit_model
        dit_model = "acestep-v15-turbo"
        print("Warning: '--model' was set to a 5Hz LM checkpoint.")
        print(f"   Treating as --lm-model: {lm_model}")
        print(f"   DiT model remains: {dit_model}")
        print()

    # ------------------------------------------------------------------ #
    # Steps default: turbo → 8, base/sft → 50                             #
    # ------------------------------------------------------------------ #
    if args.steps is None:
        if is_turbo_model(dit_model):
            args.steps = TURBO_DEFAULT_STEPS
        else:
            args.steps = NON_TURBO_DEFAULT_STEPS
        print(f"Steps auto-set to {args.steps} for model '{dit_model}'")

    # Turbo models clamp diffusion steps to 8 internally — warn early.
    if is_turbo_model(dit_model) and args.steps > TURBO_DEFAULT_STEPS:
        print(
            f"Warning: '{dit_model}' clamps --steps to {TURBO_DEFAULT_STEPS} internally; "
            f"you requested {args.steps}. Using {TURBO_DEFAULT_STEPS}."
        )
        args.steps = TURBO_DEFAULT_STEPS

    # ------------------------------------------------------------------ #
    # XL model VRAM advisory                                               #
    # ------------------------------------------------------------------ #
    if is_xl_model(dit_model):
        print(
            f"Note: XL model '{dit_model}' requires approximately "
            f"{XL_VRAM_REQUIREMENT_GB} GB of VRAM / unified memory."
        )
        print()

    # ------------------------------------------------------------------ #
    # Post-parse validation (values that argparse can't check alone)       #
    # ------------------------------------------------------------------ #
    validate_key_scale(args.key_scale)
    validate_time_signature(args.time_signature)

    # Ensure output filename ends with a recognised audio extension.
    output_path = Path(args.output)
    if output_path.suffix.lower() not in {".wav", ".flac", ".mp3", ".opus", ".aac"}:
        print(
            f"Warning: Output filename {args.output!r} has no recognised audio extension; "
            "appending '.wav'."
        )
        args.output = str(output_path.with_suffix(".wav"))

    # ------------------------------------------------------------------ #
    # Load lyrics                                                          #
    # ------------------------------------------------------------------ #
    lyrics = load_lyrics(args.lyrics_file)

    # ------------------------------------------------------------------ #
    # Start total timer                                                    #
    # ------------------------------------------------------------------ #
    total_start = time.time()

    # ------------------------------------------------------------------ #
    # Detect compute device                                                #
    # ------------------------------------------------------------------ #
    device = detect_device()

    # ------------------------------------------------------------------ #
    # Project root for model storage                                       #
    # ------------------------------------------------------------------ #
    project_root = get_project_root()

    # ------------------------------------------------------------------ #
    # Optional: ensure LM model is downloaded                             #
    # ------------------------------------------------------------------ #
    if lm_model:
        try:
            from acestep.model_downloader import ensure_lm_model

            checkpoints_root = Path(project_root) / "checkpoints"
            print(f"Ensuring 5Hz LM is available: {lm_model}")
            ok, msg = ensure_lm_model(model_name=lm_model, checkpoints_dir=checkpoints_root)
            if not ok:
                print(f"Warning: LM download/availability check failed: {msg}")
            else:
                print(f"LM ready: {msg}")
            print()
        except Exception as exc:
            print(f"Warning: Skipping LM availability check ({exc})")
            print()

    # ------------------------------------------------------------------ #
    # Initialize model                                                     #
    # ------------------------------------------------------------------ #
    handler, model_load_time, success = initialize_model(
        dit_model,
        device,
        project_root,
        prefer_source=args.download_source,
    )
    if not success:
        raise SystemExit(1)

    print(f"Model loaded in {format_duration(model_load_time)}")
    print()

    # ------------------------------------------------------------------ #
    # Print generation parameters                                          #
    # ------------------------------------------------------------------ #
    print_generation_info(args, lyrics)

    # ------------------------------------------------------------------ #
    # Generate music                                                       #
    # ------------------------------------------------------------------ #
    result, generation_time = generate_music(handler, args, lyrics)

    # ------------------------------------------------------------------ #
    # Save audio outputs                                                   #
    # ------------------------------------------------------------------ #
    save_audio_outputs(result, args.output)

    # ------------------------------------------------------------------ #
    # Display timing summary                                               #
    # ------------------------------------------------------------------ #
    if "audios" in result and result["audios"]:
        total_time = time.time() - total_start
        print_timing_summary(model_load_time, generation_time, total_time, args.duration)


if __name__ == "__main__":
    main()
