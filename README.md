# Personal Writing Style Fingerprinter

A multi-stage system to extract your unique writing voice, generate a natural-language profile, and rewrite AI-generated content in your authentic style. Train multiple voices and switch between them effortlessly.

## Overview

This project analyzes writing patterns through stylometric analysis and uses LLMs to generate human-readable voice profiles. You can then use these profiles to rewrite AI drafts, maintaining the original content while adopting a specific writer's style.

**Perfect for:**
- Content creators wanting to maintain consistent voice
- Teams collaborating on AI-written content
- Writers experimenting with different styles
- Content rewriting in multiple voices

---

## Features

✨ **Three-Stage Pipeline:**
1. **Stage 1**: Extract stylometric features (sentence length, vocabulary, punctuation, etc.)
2. **Stage 2**: Generate natural-language voice profile
3. **Stage 3**: Rewrite AI drafts in specific voices

🎭 **Multi-Voice Support:**
- Train voices from multiple writers
- Switch between voices with a simple command
- Organize and manage all voices in one place

🔄 **Flexible API Support:**
- Google Gemini (default)
- Anthropic Claude
- Groq (fastest inference)
- Easy to add more providers

📊 **Rich Analysis:**
- Sentence structure patterns
- Vocabulary diversity
- Punctuation habits
- Common phrases and sentence starters
- Readability metrics
- Tone and formality indicators

---

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/personal-voice-model.git
cd personal-voice-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API credentials:**

Choose one API provider and set the environment variable:

**For Gemini (default):**
```bash
export GOOGLE_API_KEY='your-gemini-api-key'
```

**For Claude:**
```bash
export ANTHROPIC_API_KEY='sk-your-anthropic-key'
export API_PROVIDER='anthropic'
```

**For Groq:**
```bash
export GROQ_API_KEY='your-groq-api-key'
export API_PROVIDER='groq'
```

---

## Quick Start

### Train Your Own Voice (Single User)

```bash
# Stage 1: Extract stylometric features
python stylometric_extractor.py train-data

# Stage 2: Generate voice profile
python style_profile_generator.py
# When prompted, type 'n' to skip interactive refinement

# You now have:
# - style_profile.json (features)
# - writing_style_profile.md (natural language profile)
```

### Rewrite a Draft in Your Voice

```bash
python draft_rewriter.py my_draft.txt
# Output: rewritten_my_draft.md
```

---

## Multi-Voice System (Recommended)

### Train a Friend's Voice

```bash
python voice_manager.py train friend_name path/to/their/journal/files

# Example:
python voice_manager.py train alex ~/Documents/alex_journals
```

This automatically:
1. Extracts their stylometric features
2. Generates their voice profile
3. Saves everything in `voices/alex/`

### List All Trained Voices

```bash
python voice_manager.py list

# Output:
# TRAINED VOICES
# ===============
# 📝 alex
#    Trained: 2026-04-28 14:32
#    Training files: 12
#    Location: voices/alex
```

### Rewrite a Draft in a Specific Voice

```bash
python voice_manager.py rewrite draft.md --voice alex
# Output: rewritten_alex_draft.md
```

Or specify custom output:
```bash
python voice_manager.py rewrite draft.md --voice alex --output my_version.md
```

### Batch Rewrite Multiple Drafts

```bash
# Create folder with drafts
mkdir drafts
# Add .txt or .md files to drafts/

# Batch rewrite
python voice_manager.py batch drafts --voice alex
# Output: rewrites_alex/ (folder with all rewritten versions)
```

With custom output directory:
```bash
python voice_manager.py batch drafts --voice alex --output my_rewrites
```

### Delete a Voice

```bash
python voice_manager.py delete alex
```

---

## Project Structure

```
personal-voice-model/
├── stylometric_extractor.py       # Stage 1: Feature extraction
├── style_profile_generator.py     # Stage 2: Profile generation
├── draft_rewriter.py              # Stage 3: Draft rewriting
├── voice_manager.py               # Multi-voice CLI tool
├── api_provider.py                # API abstraction layer
├── requirements.txt               # Dependencies
├── README.md                       # This file
├── sample_draft.md                # Example AI draft
├── style_profile.json             # Your extracted features
├── writing_style_profile.md       # Your voice profile
├── train-data/                    # Your journal files (for single-voice mode)
│   └── *.txt
├── voices/                        # Trained voices
│   ├── alex/
│   │   ├── train-data/
│   │   ├── style_profile.json
│   │   └── writing_style_profile.md
│   ├── sarah/
│   └── [more voices]/
└── rewrites/                      # Rewritten drafts
    └── *.md
```

---

## How It Works

### Stage 1: Stylometric Analysis

Extracts ~20 features from writing samples:
- Sentence length (avg, stdev, range)
- Vocabulary diversity (type-token ratio)
- Word length
- Punctuation patterns
- Part-of-speech distribution
- Common sentence starters
- N-grams (bigrams, trigrams)
- Readability scores (Flesch, Gunning Fog)
- Contraction usage

**Libraries used:** spaCy, NLTK, textstat

### Stage 2: Voice Profile Generation

Takes numerical features and creates a natural-language description of the writer's voice:
- Overall tone and formality
- Sentence structure preferences
- Vocabulary characteristics
- Punctuation habits
- Writing patterns and quirks

**Libraries used:** Anthropic SDK / Google Generative AI / Groq Python Client

### Stage 3: Draft Rewriting

Uses the voice profile to rewrite AI-generated content:
1. Loads the style profile (Stage 2 output)
2. Takes an AI-generated draft
3. Creates a detailed rewriting prompt
4. Rewrites the draft to match the voice
5. Preserves all original content and structure

---

## API Providers

### Gemini (Default)
- **Model**: gemini-2.0-flash
- **Cost**: Free tier available
- **Speed**: Fast
- **Setup**: Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Claude
- **Model**: claude-opus-4-6
- **Cost**: Higher but excellent quality
- **Speed**: Slower than Gemini
- **Setup**: Get key from [Anthropic Console](https://console.anthropic.com)

### Groq
- **Model**: llama-3.3-70b-versatile
- **Cost**: Free tier available
- **Speed**: Fastest inference
- **Setup**: Get key from [Groq Console](https://console.groq.com)

---

## Examples

### Example 1: Train Your Voice and Rewrite a Draft

```bash
# Step 1: Create training data folder
mkdir my_journals
# Add your journal .txt files to my_journals/

# Step 2: Extract features
python stylometric_extractor.py my_journals

# Step 3: Generate profile
python style_profile_generator.py
# Skip interactive mode: press 'n'

# Step 4: Rewrite a draft
python draft_rewriter.py ai_blog_post.md
# Output: rewritten_ai_blog_post.md
```

### Example 2: Multi-Voice Rewriting

```bash
# Train your friend's voice
python voice_manager.py train maya ~/Documents/maya_blog

# Train another friend
python voice_manager.py train jordan ~/Documents/jordan_emails

# See all voices
python voice_manager.py list

# Rewrite draft in Maya's voice
python voice_manager.py rewrite article.md --voice maya

# Rewrite same draft in Jordan's voice
python voice_manager.py rewrite article.md --voice jordan

# Batch rewrite multiple drafts
python voice_manager.py batch content_drafts --voice maya --output maya_rewrites
python voice_manager.py batch content_drafts --voice jordan --output jordan_rewrites
```

---

## Troubleshooting

### "API Key not set" Error
```bash
# Check which provider you're using
echo $API_PROVIDER  # Should be 'gemini', 'anthropic', or 'groq'

# Set the appropriate key
export GOOGLE_API_KEY='your-key'
# or
export ANTHROPIC_API_KEY='your-key'
# or
export GROQ_API_KEY='your-key'
```

### "Model decommissioned" Error (Groq)
Groq retires models periodically. Check available models at:
https://console.groq.com/docs/models

Update `api_provider.py` with the latest model name if needed.

### No .txt Files Found
Make sure your training data folder contains `.txt` files (not `.doc`, `.docx`, or other formats).

Convert to text:
```bash
# On macOS
textutil -convert txt document.docx

# Or use pandoc (install with: brew install pandoc)
pandoc document.docx -t plain -o document.txt
```

### Out of Memory
If processing very large training sets, split them into multiple smaller batches.

---

## Advanced Usage

### Custom API Provider

Add a new provider in `api_provider.py`:

```python
class YourProviderName(APIProvider):
    def __init__(self):
        self.api_key = os.getenv("YOUR_API_KEY")
        # Initialize client
    
    def is_configured(self) -> bool:
        return self.api_key is not None
    
    def generate_profile(self, system_prompt: str, user_message: str) -> str:
        # Your implementation
        pass
    
    def refine_profile(self, system_prompt: str, messages: list) -> str:
        # Your implementation
        pass
```

Then register it in `get_provider()`:
```python
elif provider_name == "yourprovider":
    provider = YourProviderName()
    # ... validation
    return provider
```

### Using Different Models

Modify the model names in `api_provider.py`:

```python
# For Gemini
model_name="gemini-2.0-flash"  # Change to another model

# For Claude
model="claude-opus-4-6"  # Change to another model

# For Groq
model="llama-3.3-70b-versatile"  # Check available models
```

---

## Contributing

Contributions welcome! Areas for improvement:
- Additional stylometric features
- More API providers
- Better voice matching algorithms
- UI/web interface
- Batch processing improvements

---

## License

MIT License - feel free to use for personal or commercial projects

---

## Questions?

Open an issue on GitHub or reach out with suggestions!

---

## Roadmap

- [ ] Web UI for easier voice training
- [ ] Voice comparison (compare two voices)
- [ ] Style transfer (blend two voices)
- [ ] Integration with popular blogging platforms
- [ ] GPU acceleration for faster processing
- [ ] Support for non-English languages
