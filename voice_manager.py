"""
Voice Manager - Manage multiple writing voices for the fingerprinter system.

Allows training new voices, rewriting with different voices, and listing trained voices.
"""

import os
import sys
import shutil
import json
import subprocess
from pathlib import Path
from datetime import datetime


VOICES_DIR = Path("voices")


def ensure_voices_dir():
    """Ensure voices directory exists."""
    VOICES_DIR.mkdir(exist_ok=True)


def voice_exists(voice_name: str) -> bool:
    """Check if a voice has been trained."""
    voice_path = VOICES_DIR / voice_name
    return voice_path.exists() and (voice_path / "writing_style_profile.md").exists()


def list_voices():
    """List all trained voices with metadata."""
    ensure_voices_dir()

    voices = []
    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            profile_file = voice_dir / "writing_style_profile.md"
            style_file = voice_dir / "style_profile.json"

            if profile_file.exists() and style_file.exists():
                # Get modification time
                mod_time = datetime.fromtimestamp(profile_file.stat().st_mtime)

                # Get training data count
                train_dir = voice_dir / "train-data"
                file_count = len(list(train_dir.glob("*.txt"))) if train_dir.exists() else 0

                voices.append({
                    "name": voice_dir.name,
                    "trained_on": mod_time.strftime("%Y-%m-%d %H:%M"),
                    "training_files": file_count,
                    "path": str(voice_dir)
                })

    return sorted(voices, key=lambda v: v["name"])


def train_voice(voice_name: str, training_data_dir: str) -> bool:
    """
    Train a new voice from writing samples.

    Args:
        voice_name: Name for the voice
        training_data_dir: Path to directory with .txt training files

    Returns:
        True if successful, False otherwise
    """
    ensure_voices_dir()

    # Validate training data directory
    if not os.path.exists(training_data_dir):
        print(f"✗ Error: Training data directory '{training_data_dir}' not found.")
        return False

    txt_files = list(Path(training_data_dir).glob("*.txt"))
    if not txt_files:
        print(f"✗ Error: No .txt files found in '{training_data_dir}'")
        return False

    print(f"Found {len(txt_files)} training files")

    # Create voice directory
    voice_dir = VOICES_DIR / voice_name
    voice_dir.mkdir(exist_ok=True)

    # Copy training data to voice directory
    train_dest = voice_dir / "train-data"
    if train_dest.exists():
        shutil.rmtree(train_dest)
    shutil.copytree(training_data_dir, train_dest)
    print(f"✓ Copied training files to {train_dest}")

    # Stage 1: Extract stylometric features
    print(f"\n📊 Stage 1: Extracting stylometric features...")
    style_profile_path = voice_dir / "style_profile.json"

    # Change to voice directory and run extractor
    original_cwd = os.getcwd()
    try:
        os.chdir(voice_dir)
        result = subprocess.run(
            [sys.executable, "../../stylometric_extractor.py", "train-data"],
            capture_output=True,
            text=True
        )
        os.chdir(original_cwd)

        if result.returncode != 0:
            print(f"✗ Error during feature extraction:")
            print(result.stderr)
            return False

        if not style_profile_path.exists():
            print(f"✗ Error: style_profile.json not created")
            return False

        print(f"✓ Stylometric features extracted")
    except Exception as e:
        os.chdir(original_cwd)
        print(f"✗ Error: {e}")
        return False

    # Stage 2: Generate natural language profile
    print(f"\n🎨 Stage 2: Generating voice profile...")

    # Check for API key
    api_provider = os.getenv("API_PROVIDER", "gemini").lower()
    if api_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print(f"✗ Error: ANTHROPIC_API_KEY not set")
            return False
    elif api_provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            print(f"✗ Error: GOOGLE_API_KEY not set")
            return False
    elif api_provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            print(f"✗ Error: GROQ_API_KEY not set")
            return False

    try:
        os.chdir(voice_dir)
        result = subprocess.run(
            [sys.executable, "../../style_profile_generator.py"],
            capture_output=True,
            text=True,
            input="n\n"  # Skip interactive refinement
        )
        os.chdir(original_cwd)

        if result.returncode != 0:
            print(f"✗ Error during profile generation:")
            print(result.stderr)
            return False

        profile_file = voice_dir / "writing_style_profile.md"
        if not profile_file.exists():
            print(f"✗ Error: writing_style_profile.md not created")
            return False

        print(f"✓ Voice profile generated")
    except Exception as e:
        os.chdir(original_cwd)
        print(f"✗ Error: {e}")
        return False

    print(f"\n✅ Voice '{voice_name}' trained successfully!")
    print(f"   Location: {voice_dir}")
    return True


def rewrite_with_voice(draft_file: str, voice_name: str, output_file: str = None) -> bool:
    """
    Rewrite a draft using a specific voice.

    Args:
        draft_file: Path to draft to rewrite
        voice_name: Name of the voice to use
        output_file: Optional output file path

    Returns:
        True if successful, False otherwise
    """
    # Check if draft exists
    if not os.path.exists(draft_file):
        print(f"✗ Error: Draft file '{draft_file}' not found.")
        return False

    # Check if voice exists
    if not voice_exists(voice_name):
        print(f"✗ Error: Voice '{voice_name}' not trained.")
        print(f"   Available voices: {', '.join(v['name'] for v in list_voices())}")
        return False

    voice_dir = VOICES_DIR / voice_name
    profile_file = voice_dir / "writing_style_profile.md"

    # Load style profile
    with open(profile_file, "r") as f:
        style_profile = f.read()

    # Load draft
    with open(draft_file, "r") as f:
        draft_text = f.read()

    # Get API provider
    try:
        from api_provider import get_provider
        provider = get_provider()
    except EnvironmentError as e:
        print(f"✗ Error: {e}")
        return False

    # Rewrite
    print(f"\n🎭 Rewriting with voice: {voice_name}...")

    from draft_rewriter import create_rewriting_prompt, rewrite_draft

    try:
        rewritten = rewrite_draft(provider, draft_text, style_profile)
    except Exception as e:
        print(f"✗ Error during rewriting: {e}")
        return False

    # Save output
    if output_file is None:
        base_path = Path(draft_file)
        output_file = f"rewritten_{voice_name}_{base_path.stem}.md"

    with open(output_file, "w") as f:
        f.write(rewritten)

    print(f"✓ Rewritten draft saved to: {output_file}")
    return True


def batch_rewrite_with_voice(drafts_dir: str, voice_name: str, output_dir: str = None) -> dict:
    """
    Batch rewrite multiple drafts with a specific voice.

    Args:
        drafts_dir: Directory containing drafts
        voice_name: Name of voice to use
        output_dir: Optional output directory

    Returns:
        Results dictionary
    """
    # Check if voice exists
    if not voice_exists(voice_name):
        print(f"✗ Error: Voice '{voice_name}' not trained.")
        return {}

    # Find draft files
    draft_files = list(Path(drafts_dir).glob("*.txt")) + list(Path(drafts_dir).glob("*.md"))

    if not draft_files:
        print(f"No draft files found in {drafts_dir}")
        return {}

    # Create output directory
    if output_dir is None:
        output_dir = f"rewrites_{voice_name}"

    Path(output_dir).mkdir(exist_ok=True)

    print(f"\nBatch rewriting {len(draft_files)} drafts with voice: {voice_name}...")

    results = {}
    for draft_path in draft_files:
        output_file = Path(output_dir) / f"rewritten_{voice_name}_{draft_path.stem}.md"
        success = rewrite_with_voice(str(draft_path), voice_name, str(output_file))
        results[draft_path.name] = "success" if success else "error"

    return results


def show_help():
    """Show help message."""
    help_text = """
╔════════════════════════════════════════════════════════════════════╗
║                      VOICE MANAGER HELP                           ║
╚════════════════════════════════════════════════════════════════════╝

USAGE:
  python voice_manager.py <command> [arguments]

COMMANDS:

1. LIST TRAINED VOICES
   python voice_manager.py list
   python voice_manager.py ls

2. TRAIN A NEW VOICE
   python voice_manager.py train <voice_name> <training_data_dir>

   Example:
   python voice_manager.py train john_doe path/to/john/journal_files
   python voice_manager.py train friend_mary ~/Documents/mary_writings

3. REWRITE A DRAFT WITH A VOICE
   python voice_manager.py rewrite <draft_file> --voice <voice_name>
   python voice_manager.py rewrite <draft_file> --voice <voice_name> --output <output_file>

   Example:
   python voice_manager.py rewrite sample_draft.md --voice john_doe
   python voice_manager.py rewrite blog_post.txt --voice friend_mary --output my_version.md

4. BATCH REWRITE (Multiple Drafts)
   python voice_manager.py batch <drafts_dir> --voice <voice_name>
   python voice_manager.py batch <drafts_dir> --voice <voice_name> --output <output_dir>

   Example:
   python voice_manager.py batch drafts --voice john_doe
   python voice_manager.py batch my_drafts --voice friend_mary --output my_rewrites

5. DELETE A VOICE
   python voice_manager.py delete <voice_name>

ENVIRONMENT VARIABLES:
  API_PROVIDER    Which API to use (gemini, anthropic, groq). Default: gemini
  GOOGLE_API_KEY  For Gemini API
  ANTHROPIC_API_KEY  For Claude API
  GROQ_API_KEY    For Groq API

EXAMPLES:

# Train your friend's voice
python voice_manager.py train alex ~/journal_files/alex

# See all trained voices
python voice_manager.py list

# Rewrite a draft in your friend's voice
python voice_manager.py rewrite draft.md --voice alex

# Batch rewrite multiple drafts
python voice_manager.py batch my_drafts --voice alex --output rewrites_alex
"""
    print(help_text)


def delete_voice(voice_name: str) -> bool:
    """Delete a trained voice."""
    voice_dir = VOICES_DIR / voice_name

    if not voice_dir.exists():
        print(f"✗ Error: Voice '{voice_name}' not found.")
        return False

    confirm = input(f"⚠ Delete voice '{voice_name}'? This cannot be undone. (y/n): ").strip().lower()

    if confirm != "y":
        print("Cancelled.")
        return False

    shutil.rmtree(voice_dir)
    print(f"✓ Voice '{voice_name}' deleted.")
    return True


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command in ["list", "ls"]:
        voices = list_voices()
        if not voices:
            print("No trained voices found.")
            return

        print("\n" + "="*70)
        print("TRAINED VOICES")
        print("="*70)
        for voice in voices:
            print(f"\n📝 {voice['name']}")
            print(f"   Trained: {voice['trained_on']}")
            print(f"   Training files: {voice['training_files']}")
            print(f"   Location: {voice['path']}")
        print("\n" + "="*70 + "\n")

    elif command == "train":
        if len(sys.argv) < 4:
            print("Usage: python voice_manager.py train <voice_name> <training_data_dir>")
            return

        voice_name = sys.argv[2]
        training_dir = sys.argv[3]

        if voice_exists(voice_name):
            confirm = input(f"⚠ Voice '{voice_name}' already exists. Retrain? (y/n): ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return

        print(f"\n🎓 Training voice: {voice_name}")
        print(f"   Training data: {training_dir}\n")

        success = train_voice(voice_name, training_dir)
        if success:
            print(f"\n✅ You can now use this voice:")
            print(f"   python voice_manager.py rewrite draft.md --voice {voice_name}")

    elif command == "rewrite":
        if len(sys.argv) < 2:
            print("Usage: python voice_manager.py rewrite <draft_file> --voice <voice_name> [--output <output_file>]")
            return

        draft_file = sys.argv[2]
        voice_name = None
        output_file = None

        # Parse arguments
        for i in range(3, len(sys.argv)):
            if sys.argv[i] == "--voice" and i + 1 < len(sys.argv):
                voice_name = sys.argv[i + 1]
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]

        if not voice_name:
            print("Error: --voice argument required")
            print("Usage: python voice_manager.py rewrite <draft_file> --voice <voice_name>")
            return

        rewrite_with_voice(draft_file, voice_name, output_file)

    elif command == "batch":
        if len(sys.argv) < 3:
            print("Usage: python voice_manager.py batch <drafts_dir> --voice <voice_name> [--output <output_dir>]")
            return

        drafts_dir = sys.argv[2]
        voice_name = None
        output_dir = None

        # Parse arguments
        for i in range(3, len(sys.argv)):
            if sys.argv[i] == "--voice" and i + 1 < len(sys.argv):
                voice_name = sys.argv[i + 1]
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]

        if not voice_name:
            print("Error: --voice argument required")
            return

        results = batch_rewrite_with_voice(drafts_dir, voice_name, output_dir)

        if results:
            success = sum(1 for r in results.values() if r == "success")
            print(f"\n✓ Batch rewriting complete: {success}/{len(results)} successful")

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python voice_manager.py delete <voice_name>")
            return

        voice_name = sys.argv[2]
        delete_voice(voice_name)

    elif command in ["help", "-h", "--help"]:
        show_help()

    else:
        print(f"Unknown command: {command}")
        print("Run 'python voice_manager.py help' for usage information.")


if __name__ == "__main__":
    main()
