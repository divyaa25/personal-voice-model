"""
Personal Writing Style Fingerprinter - Stage 2: Natural Language Style Profile Generator

Takes the stylometric features extracted in Stage 1 and uses Claude/Gemini to generate
a human-readable style profile document describing your unique voice.

Supports multiple API providers via API_PROVIDER env variable:
- 'gemini' (default, requires GOOGLE_API_KEY)
- 'anthropic' (requires ANTHROPIC_API_KEY)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from api_provider import get_provider


def load_style_profile(filepath: str = "style_profile.json") -> dict:
    """Load the stylometric profile from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Profile file '{filepath}' not found. Run stylometric_extractor.py first."
        )

    with open(filepath, "r") as f:
        return json.load(f)


def format_profile_for_claude(profile: dict) -> str:
    """Format the stylometric data into a readable prompt for Claude."""

    # Helper to format lists of tuples (like sentence starters)
    def format_list_of_tuples(items, max_items=10):
        if not items:
            return "None"
        formatted = []
        for item, count in items[:max_items]:
            formatted.append(f"- {item} (frequency: {count})")
        return "\n".join(formatted)

    # Helper to format dict
    def format_dict(d, max_items=10):
        if not d:
            return "None"
        items = list(d.items())[:max_items]
        return "\n".join(f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}"
                        for k, v in items)

    prompt = f"""
# STYLOMETRIC DATA EXTRACTION

## Overview Statistics
- Total files analyzed: {profile.get('file_count', 'N/A')}
- Average word count per document: {profile.get('avg_word_count', 0):.1f}
- Average sentences per document: {profile.get('avg_sentence_count', 0):.1f}
- Average paragraphs per document: {profile.get('avg_paragraph_count', 0):.1f}

## Sentence Structure
- Average sentence length: {profile.get('avg_avg_sentence_length', 0):.2f} words
- Sentence length variation (stdev): {profile.get('stdev_avg_sentence_length', 0):.2f}

## Vocabulary Characteristics
- Type-Token Ratio (vocabulary diversity): {profile.get('avg_type_token_ratio', 0):.3f}
- Average unique words per document: {profile.get('avg_unique_word_count', 0):.0f}
- Average word length: {profile.get('avg_avg_word_length', 0):.2f} characters
- Content word ratio (substantive vs. filler): {profile.get('avg_content_word_ratio', 0):.3f}
- Contraction usage ratio: {profile.get('avg_contraction_ratio', 0):.3f}

## Readability Metrics
- Flesch-Kincaid Grade Level: {profile.get('avg_flesch_kincaid_grade', 0):.1f}
- Flesch Reading Ease: {profile.get('avg_flesch_reading_ease', 0):.1f} (0-100 scale; higher = easier)
- Gunning Fog Index: {profile.get('avg_gunning_fog', 0):.1f}

## Punctuation Habits (average per document)
{format_dict(profile.get('avg_punctuation', {}))}

## Part of Speech Distribution
{format_dict(profile.get('avg_pos_distribution', {}))}

## Common Sentence Starters
{format_list_of_tuples(profile.get('top_sentence_starters', []), max_items=10)}

## Most Common Phrases (Bigrams)
{format_list_of_tuples(profile.get('top_bigrams', []), max_items=10)}

## Most Common Phrases (Trigrams)
{format_list_of_tuples(profile.get('top_trigrams', []), max_items=10)}
"""
    return prompt


def generate_style_profile(provider, profile: dict, conversation_history: list = None) -> tuple[str, list]:
    """
    Use the configured API provider to generate a natural-language style profile.

    Returns:
        Tuple of (generated_profile_text, updated_conversation_history)
    """

    if conversation_history is None:
        conversation_history = []

    # Format the data
    stylometric_data = format_profile_for_claude(profile)

    # System prompt
    system_prompt = """You are an expert writing analyst and literary style guide.
Your task is to take stylometric data from a writer's text and generate a comprehensive,
natural-language style profile that captures their unique voice, habits, and characteristics.

The profile should:
1. Describe their overall writing voice and tone (formal, conversational, etc.)
2. Analyze their sentence structure preferences
3. Discuss their vocabulary choices and diversity
4. Explain their punctuation habits and what they suggest
5. Identify common patterns and recurring phrases
6. Note what their readability scores suggest about their writing
7. Provide actionable insights for understanding their style
8. Be written as a cohesive narrative, not just bullet points

Make the profile engaging, insightful, and specific to the data provided."""

    # Generate initial profile
    if not conversation_history:
        user_message = f"Here's my stylometric data from my recent writing. Please analyze it and create a comprehensive natural-language style profile:\n\n{stylometric_data}"

        profile_text = provider.generate_profile(system_prompt, user_message)

        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        conversation_history.append({
            "role": "assistant",
            "content": profile_text
        })

    return profile_text, conversation_history


def refine_profile_with_context(provider, conversation_history: list, user_input: str) -> tuple[str, list]:
    """
    Allow the user to ask follow-up questions to refine their profile.
    """

    system_prompt = """You are an expert writing analyst. You're helping a writer understand their
unique voice and style. Answer follow-up questions about their writing characteristics,
offer suggestions for how to maintain consistency, or provide guidance on how to apply
these insights to their writing."""

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    refined_text = provider.refine_profile(system_prompt, conversation_history)

    conversation_history.append({
        "role": "assistant",
        "content": refined_text
    })

    return refined_text, conversation_history


def save_profile_document(profile_text: str, filename: str = "writing_style_profile.md"):
    """Save the generated profile as a markdown document."""

    content = f"""# Your Personal Writing Style Profile

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{profile_text}

---

*This profile was automatically generated from analysis of your recent writing samples using stylometric analysis and Claude AI. Use this as a reference when writing new content to maintain consistency with your natural voice.*
"""

    with open(filename, "w") as f:
        f.write(content)

    print(f"\n✓ Profile saved to {filename}")
    return filename


def interactive_refinement(provider, conversation_history: list):
    """Interactive mode for asking follow-up questions about your style."""

    print("\n" + "="*70)
    print("STYLE PROFILE REFINEMENT MODE")
    print("="*70)
    print("You can now ask follow-up questions about your writing style.")
    print("Type 'exit' or 'quit' to finish.\n")

    while True:
        user_input = input("Your question (or 'exit'): ").strip()

        if user_input.lower() in ["exit", "quit", ""]:
            break

        print("\nAnalyzing...")
        response, conversation_history = refine_profile_with_context(
            provider, conversation_history, user_input
        )
        print(f"\n{response}\n")

    return conversation_history


def main():
    """Main pipeline for Stage 2."""

    print("="*70)
    print("STAGE 2: NATURAL LANGUAGE STYLE PROFILE GENERATOR")
    print("="*70)

    # Initialize API provider based on environment
    provider_name = os.getenv("API_PROVIDER", "gemini").lower()
    print(f"\nUsing API provider: {provider_name}")

    try:
        provider = get_provider()
    except EnvironmentError as e:
        print(f"\n⚠ Error: {e}")
        return

    # Load the profile from Stage 1
    print("\nLoading stylometric profile...")
    try:
        profile = load_style_profile()
        print(f"✓ Loaded profile from {profile.get('file_count', 0)} files")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return

    # Generate the style profile with the configured API
    print(f"\nGenerating natural-language profile...")
    profile_text, conversation_history = generate_style_profile(provider, profile)

    # Save to markdown
    save_profile_document(profile_text)

    # Print a preview
    print("\n" + "="*70)
    print("GENERATED PROFILE PREVIEW")
    print("="*70)
    print(profile_text[:1000] + "..." if len(profile_text) > 1000 else profile_text)

    # Offer interactive refinement
    print("\n" + "="*70)
    refine = input("\nWould you like to ask follow-up questions about your style? (y/n): ").strip().lower()

    if refine == "y":
        conversation_history = interactive_refinement(provider, conversation_history)

        # Save the refined version
        print("\nSaving final refined profile...")
        final_text = "\n\n".join([
            profile_text,
            "\n## Refinements & Follow-ups\n",
            "\n\n".join([
                msg["content"] for msg in conversation_history[1:]
                if msg["role"] == "assistant"
            ])
        ])
        save_profile_document(final_text, "writing_style_profile_final.md")

    print("\n✓ Stage 2 complete!")
    print("Your style profile is ready for Stage 3 (rewriting AI drafts in your voice).")


if __name__ == "__main__":
    main()
