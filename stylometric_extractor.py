"""
Personal Writing Style Fingerprinter - Stage 1: Stylometric Feature Extraction

This script ingests .txt journal files and extracts key stylometric features
to build a profile of your natural writing voice.
"""

import os
import re
from pathlib import Path
from collections import Counter
import statistics
from typing import Dict, List, Tuple

import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class StyleometricExtractor:
    """Extract stylometric features from text."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_from_file(self, filepath: str) -> Dict:
        """Extract all stylometric features from a single file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.extract_from_text(text)

    def extract_from_text(self, text: str) -> Dict:
        """Extract stylometric features from raw text."""
        if not text.strip():
            return {}

        # Basic counts
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_no_punct = [w for w in words if w.isalnum()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Process with spaCy for deeper analysis
        doc = nlp(text)

        features = {
            # Sentence-level features
            'sentence_count': len(sentences),
            'avg_sentence_length': self._avg_sentence_length(sentences),
            'sentence_length_stdev': self._sentence_length_stdev(sentences),
            'sentence_length_range': self._sentence_length_range(sentences),

            # Word-level features
            'word_count': len(words_no_punct),
            'unique_word_count': len(set(words_no_punct)),
            'type_token_ratio': len(set(words_no_punct)) / len(words_no_punct) if words_no_punct else 0,
            'avg_word_length': statistics.mean(len(w) for w in words_no_punct) if words_no_punct else 0,

            # Vocabulary features
            'lexical_diversity': self._lexical_diversity(words_no_punct),
            'content_word_ratio': self._content_word_ratio(words_no_punct),

            # Punctuation habits
            'punctuation_profile': self._punctuation_profile(text),

            # Paragraph structure
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': len(words_no_punct) / len(paragraphs) if paragraphs else 0,

            # Sentence starters
            'sentence_starters': self._extract_sentence_starters(sentences),

            # Common phrases (bigrams and trigrams)
            'common_bigrams': self._extract_ngrams(words_no_punct, 2, top_n=10),
            'common_trigrams': self._extract_ngrams(words_no_punct, 3, top_n=10),

            # Readability scores
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'gunning_fog': textstat.gunning_fog(text),

            # Part of speech patterns
            'pos_distribution': self._pos_distribution(doc),

            # Contraction and formality indicators
            'contraction_ratio': self._contraction_ratio(text, words_no_punct),
        }

        return features

    def _avg_sentence_length(self, sentences: List[str]) -> float:
        """Calculate average sentence length in words."""
        if not sentences:
            return 0
        lengths = [len(word_tokenize(s)) for s in sentences]
        return statistics.mean(lengths)

    def _sentence_length_stdev(self, sentences: List[str]) -> float:
        """Calculate standard deviation of sentence lengths."""
        if len(sentences) < 2:
            return 0
        lengths = [len(word_tokenize(s)) for s in sentences]
        return statistics.stdev(lengths)

    def _sentence_length_range(self, sentences: List[str]) -> Dict[str, int]:
        """Get min and max sentence lengths."""
        if not sentences:
            return {'min': 0, 'max': 0}
        lengths = [len(word_tokenize(s)) for s in sentences]
        return {'min': min(lengths), 'max': max(lengths)}

    def _lexical_diversity(self, words: List[str]) -> float:
        """Calculate lexical diversity using MTLD-style metric."""
        # Simplified measure: unique words / total words (already captured as type_token_ratio)
        # This is a placeholder for more sophisticated metrics
        return len(set(words)) / len(words) if words else 0

    def _content_word_ratio(self, words: List[str]) -> float:
        """Calculate ratio of content words (non-stopwords) to total words."""
        content_words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return len(content_words) / len(words) if words else 0

    def _punctuation_profile(self, text: str) -> Dict[str, int]:
        """Extract punctuation habits."""
        profile = {
            'periods': text.count('.'),
            'commas': text.count(','),
            'semicolons': text.count(';'),
            'colons': text.count(':'),
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'dashes': text.count('-'),
            'ellipsis': text.count('...'),
            'parentheses': text.count('('),
            'quotes': text.count('"'),
        }
        return profile

    def _extract_sentence_starters(self, sentences: List[str], top_n: int = 15) -> List[Tuple[str, int]]:
        """Extract the most common sentence starters (first 1-3 words)."""
        starters = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            if words:
                # Get first 1-3 words as starter
                starter = ' '.join(words[:min(3, len(words))])
                starters.append(starter)

        return Counter(starters).most_common(top_n)

    def _extract_ngrams(self, words: List[str], n: int, top_n: int = 10) -> List[Tuple[str, int]]:
        """Extract most common n-grams."""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)

        return Counter(ngrams).most_common(top_n)

    def _pos_distribution(self, doc) -> Dict[str, float]:
        """Calculate distribution of parts of speech."""
        pos_counts = Counter(token.pos_ for token in doc)
        total = len(doc)
        if total == 0:
            return {}
        return {pos: count / total for pos, count in pos_counts.most_common(10)}

    def _contraction_ratio(self, text: str, words: List[str]) -> float:
        """Calculate ratio of contractions (e.g., don't, it's)."""
        contraction_pattern = re.compile(r"\b\w+n't\b|\b\w+'[sd]\b|\b\w+'ll\b|\b\w+'ve\b|\b\w+'re\b")
        contractions = len(contraction_pattern.findall(text))
        return contractions / len(words) if words else 0


def batch_extract(journal_dir: str) -> Dict:
    """
    Extract stylometric features from all .txt files in a directory.

    Args:
        journal_dir: Path to directory containing .txt journal files

    Returns:
        Dictionary with aggregated statistics across all files
    """
    extractor = StyleometricExtractor()
    all_files = list(Path(journal_dir).glob('*.txt'))

    if not all_files:
        print(f"No .txt files found in {journal_dir}")
        return {}

    print(f"Processing {len(all_files)} files...")

    all_features = []
    for filepath in all_files:
        print(f"  Processing {filepath.name}...")
        features = extractor.extract_from_file(str(filepath))
        if features:
            all_features.append(features)

    if not all_features:
        print("No features extracted.")
        return {}

    # Aggregate features across all files
    aggregated = _aggregate_features(all_features)
    aggregated['file_count'] = len(all_files)
    aggregated['total_files_processed'] = len(all_files)

    return aggregated


def _aggregate_features(features_list: List[Dict]) -> Dict:
    """Aggregate features across multiple documents."""
    aggregated = {}

    # Numeric features to average
    numeric_features = [
        'sentence_count', 'avg_sentence_length', 'sentence_length_stdev',
        'word_count', 'unique_word_count', 'type_token_ratio', 'avg_word_length',
        'lexical_diversity', 'content_word_ratio', 'paragraph_count',
        'avg_paragraph_length', 'flesch_kincaid_grade', 'flesch_reading_ease',
        'gunning_fog', 'contraction_ratio'
    ]

    for feature in numeric_features:
        values = [f.get(feature, 0) for f in features_list if feature in f]
        if values:
            aggregated[f'avg_{feature}'] = statistics.mean(values)
            aggregated[f'stdev_{feature}'] = statistics.stdev(values) if len(values) > 1 else 0

    # Aggregate sentence starters
    all_starters = []
    for f in features_list:
        all_starters.extend(f.get('sentence_starters', []))
    if all_starters:
        aggregated['top_sentence_starters'] = Counter(
            [starter for starter, _ in all_starters]
        ).most_common(10)

    # Aggregate n-grams
    all_bigrams = []
    all_trigrams = []
    for f in features_list:
        all_bigrams.extend(f.get('common_bigrams', []))
        all_trigrams.extend(f.get('common_trigrams', []))

    if all_bigrams:
        aggregated['top_bigrams'] = Counter(
            [bigram for bigram, _ in all_bigrams]
        ).most_common(10)

    if all_trigrams:
        aggregated['top_trigrams'] = Counter(
            [trigram for trigram, _ in all_trigrams]
        ).most_common(10)

    # Aggregate punctuation
    punct_profiles = [f.get('punctuation_profile', {}) for f in features_list]
    if punct_profiles:
        aggregated['avg_punctuation'] = {}
        for key in punct_profiles[0].keys():
            values = [p.get(key, 0) for p in punct_profiles]
            aggregated['avg_punctuation'][key] = statistics.mean(values)

    # Aggregate POS distribution
    pos_dists = [f.get('pos_distribution', {}) for f in features_list]
    if pos_dists:
        all_pos = {}
        for pos_dist in pos_dists:
            for pos, ratio in pos_dist.items():
                all_pos[pos] = all_pos.get(pos, 0) + ratio
        aggregated['avg_pos_distribution'] = {
            pos: ratio / len(pos_dists) for pos, ratio in all_pos.items()
        }

    return aggregated


def print_style_profile(features: Dict):
    """Pretty-print the extracted stylometric features."""
    print("\n" + "=" * 70)
    print("STYLOMETRIC PROFILE")
    print("=" * 70)

    sections = {
        'Overview': [
            'file_count', 'total_files_processed',
            'avg_word_count', 'avg_sentence_count', 'avg_paragraph_count'
        ],
        'Sentence Characteristics': [
            'avg_avg_sentence_length', 'stdev_avg_sentence_length',
            'avg_sentence_length_range'
        ],
        'Vocabulary & Diversity': [
            'avg_type_token_ratio', 'avg_lexical_diversity',
            'avg_unique_word_count', 'avg_avg_word_length',
            'avg_content_word_ratio'
        ],
        'Readability': [
            'avg_flesch_kincaid_grade', 'avg_flesch_reading_ease',
            'avg_gunning_fog'
        ],
        'Tone & Formality': [
            'avg_contraction_ratio', 'avg_pos_distribution'
        ],
        'Punctuation Habits': [
            'avg_punctuation'
        ],
        'Common Patterns': [
            'top_sentence_starters', 'top_bigrams', 'top_trigrams'
        ],
    }

    for section_name, keys in sections.items():
        section_data = {k: features[k] for k in keys if k in features}
        if section_data:
            print(f"\n{section_name}:")
            print("-" * 70)
            for key, value in section_data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, list) and value and isinstance(value[0], tuple):
                    # Format list of (item, count) tuples
                    print(f"  {key}:")
                    for item, count in value[:5]:
                        print(f"    - {item}: {count}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in list(value.items())[:5]:
                        if isinstance(v, float):
                            print(f"    - {k}: {v:.2f}")
                        else:
                            print(f"    - {k}: {v}")
                else:
                    print(f"  {key}: {value}")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    import json
    import sys

    # Get directory from command line or use default
    journal_dir = sys.argv[1] if len(sys.argv) > 1 else './journal_samples'

    # Check if directory exists
    if not os.path.exists(journal_dir):
        print(f"Error: Directory '{journal_dir}' does not exist.")
        print(f"Usage: python stylometric_extractor.py <path_to_journal_dir>")
        sys.exit(1)

    # Extract features
    features = batch_extract(journal_dir)

    # Print formatted profile
    if features:
        print_style_profile(features)

        # Save as JSON for later use (Stage 2)
        output_file = 'style_profile.json'
        with open(output_file, 'w') as f:
            # Convert Counter objects to dicts for JSON serialization
            features_serializable = {}
            for k, v in features.items():
                if isinstance(v, Counter):
                    features_serializable[k] = dict(v)
                else:
                    features_serializable[k] = v

            json.dump(features_serializable, f, indent=2)
            print(f"Profile saved to {output_file}")
