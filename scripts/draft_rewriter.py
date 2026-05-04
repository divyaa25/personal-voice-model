"""
Personal Writing Style Fingerprinter - Stage 3: AI Draft Rewriter

Takes AI-generated blog post drafts and rewrites them in your unique voice
using the style profile generated in Stage 2.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from api_provider import get_provider


def load_style_profile(filepath: str = "writing_style_profile.md") -> str:
    """Load the style profile document."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Style profile '{filepath}' not found. Run stage_profile_generator.py first."
        )

    with open(filepath, "r") as f:
        return f.read()


def load_draft(filepath: str) -> str:
    """Load an AI-generated draft."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Draft file '{filepath}' not found.")

    with open(filepath, "r") as f:
        return f.read()


def create_rewriting_prompt(draft: str, style_profile: str) -> str:
    """Create a prompt for rewriting the draft in the user's voice."""
    return f"""# TASK: Rewrite this draft in my unique voice

## MY WRITING STYLE PROFILE
{style_profile}

---

## ORIGINAL DRAFT TO REWRITE
{draft}

---

## INSTRUCTIONS

Please rewrite the above draft in MY voice, maintaining:
1. The core structure and key points from the original
2. All factual information and arguments
3. The same general length and section breaks

But transform it to match my unique writing style:
- Use my characteristic sentence starters and phrasing patterns
- Match my vocabulary diversity and word choices
- Adopt my tone and formality level
- Mirror my punctuation habits (frequency of commas, colons, dashes, etc.)
- Incorporate my common phrases and verbal patterns
- Maintain my readability level and sentence complexity
- Use my preferred level of contractions and conversational tone

The rewritten version should feel like I wrote it, not like an AI translation.
Preserve the original meaning but make it sound authentically like me."""


def rewrite_draft(provider, draft: str, style_profile: str) -> str:
    """Rewrite an AI draft in the user's voice using their style profile."""
    prompt = create_rewriting_prompt(draft, style_profile)

    system_message = """You are an expert writing coach specializing in voice and style.
Your task is to rewrite content to match a specific writer's unique voice while preserving
all the original information and structure. Make the rewritten version sound like it was
written by the person whose style profile you've been given. Focus on authenticity and
naturalness rather than literal matching."""

    rewritten = provider.generate_profile(system_message, prompt)
    return rewritten


def batch_rewrite_drafts(drafts_dir: str, output_dir: str = "rewrites") -> dict:
    """
    Rewrite all AI drafts in a directory.

    Args:
        drafts_dir: Directory containing .txt or .md draft files
        output_dir: Directory to save rewritten versions

    Returns:
        Dictionary with results for each file
    """
    # Get API provider
    try:
        provider = get_provider()
    except EnvironmentError as e:
        print(f"⚠ Error: {e}")
        return {}

    # Load style profile
    print("\nLoading your style profile...")
    try:
        style_profile = load_style_profile()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return {}

    # Find draft files
    draft_files = list(Path(drafts_dir).glob("*.txt")) + list(Path(drafts_dir).glob("*.md"))

    if not draft_files:
        print(f"No draft files found in {drafts_dir}")
        return {}

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\nRewriting {len(draft_files)} drafts...")
    results = {}

    for draft_path in draft_files:
        print(f"\n  Processing: {draft_path.name}...", end=" ", flush=True)

        try:
            # Load draft
            draft_text = load_draft(str(draft_path))

            # Rewrite in user's voice
            rewritten = rewrite_draft(provider, draft_text, style_profile)

            # Save rewritten version
            output_name = f"rewritten_{draft_path.stem}.md"
            output_path = Path(output_dir) / output_name

            with open(output_path, "w") as f:
                f.write(rewritten)

            results[draft_path.name] = {
                "status": "success",
                "output_file": str(output_path),
                "original_chars": len(draft_text),
                "rewritten_chars": len(rewritten)
            }

            print("✓")

        except Exception as e:
            results[draft_path.name] = {
                "status": "error",
                "error": str(e)
            }
            print(f"✗ ({str(e)[:50]}...)")

    return results


def single_draft_rewrite(draft_filepath: str, output_filepath: str = None) -> str:
    """Rewrite a single draft file."""
    # Get API provider
    try:
        provider = get_provider()
    except EnvironmentError as e:
        print(f"⚠ Error: {e}")
        return ""

    # Load style profile
    print("Loading your style profile...")
    try:
        style_profile = load_style_profile()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return ""

    # Load draft
    print(f"Loading draft: {draft_filepath}...")
    try:
        draft_text = load_draft(draft_filepath)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return ""

    # Rewrite
    print("\nRewriting in your voice...")
    rewritten = rewrite_draft(provider, draft_text, style_profile)

    # Save output
    if output_filepath is None:
        base_path = Path(draft_filepath)
        output_filepath = f"rewritten_{base_path.stem}.md"

    with open(output_filepath, "w") as f:
        f.write(rewritten)

    print(f"✓ Rewritten draft saved to: {output_filepath}")
    return rewritten


def main():
    """Main pipeline for Stage 3."""

    print("="*70)
    print("STAGE 3: AI DRAFT REWRITER")
    print("="*70)

    import sys

    if len(sys.argv) > 1:
        # Single file mode
        draft_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"\n📝 Rewriting single draft: {draft_file}\n")
        single_draft_rewrite(draft_file, output_file)

    else:
        # Batch mode
        drafts_dir = "drafts"

        if not os.path.exists(drafts_dir):
            print(f"\n⚠ Error: '{drafts_dir}' directory not found.")
            print("Create a 'drafts' folder with your AI-generated .txt or .md files.")
            print("\nOr rewrite a single file:")
            print("  python draft_rewriter.py <path_to_draft.txt>")
            print("  python draft_rewriter.py <path_to_draft.txt> <output_path.md>")
            return

        results = batch_rewrite_drafts(drafts_dir)

        if results:
            print("\n" + "="*70)
            print("REWRITING COMPLETE")
            print("="*70)

            success_count = sum(1 for r in results.values() if r["status"] == "success")
            error_count = sum(1 for r in results.values() if r["status"] == "error")

            print(f"\n✓ Successfully rewritten: {success_count}/{len(results)}")
            if error_count > 0:
                print(f"✗ Failed: {error_count}")

            print("\nRewritten drafts saved to: ./rewrites/")
            print("\n📊 Details:")
            for filename, result in results.items():
                if result["status"] == "success":
                    print(f"  ✓ {filename}")
                    print(f"    → {result['output_file']}")
                    print(f"    Original: {result['original_chars']} chars | Rewritten: {result['rewritten_chars']} chars")
                else:
                    print(f"  ✗ {filename}: {result['error']}")

            print("\n" + "="*70)


if __name__ == "__main__":
    main()
