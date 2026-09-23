# -*- coding: utf-8 -*-
"""
data_preprocessing.py
---------------------
Preprocessing module for Experiment 8: Grammar Error Correction & Text Rewriting.
Cleans raw text, standardizes whitespace, normalizes unicode punctuation and quotes,
segments sentences, extracts length/vocabulary metrics, and verifies data integrity.
"""

import re
import html
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def clean_text(text: str) -> str:
    """
    Standardizes encoding, removes HTML tags/entities, normalizes smart quotes,
    dashes, and trims extraneous whitespace.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Unescape HTML entities & strip tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Normalize smart quotes and typographical dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u00A0', ' ')

    # Consolidate whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_punctuation(text: str) -> str:
    """
    Standardizes spacing around common punctuation marks while preserving decimal numbers.
    """
    text = clean_text(text)
    # Remove space before standard punctuation marks
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    # Ensure a space after punctuation when followed immediately by an alphabetic character
    text = re.sub(r'([,.:;?!])([A-Za-z])', r'\1 \2', text)
    return text.strip()


def tokenize_sentences(text: str) -> List[str]:
    """Splits multi-sentence paragraphs into individual sentences cleanly."""
    text = clean_text(text)
    if not text:
        return []
    # Splitting on period/exclamation/question mark followed by space or end
    raw_sents = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in raw_sents if s.strip()]


def compute_text_statistics(text: str) -> Dict[str, Any]:
    """Computes word count, character count, sentence count, and average word length."""
    cleaned = clean_text(text)
    words = re.findall(r'\b\w+\b', cleaned)
    chars = len(cleaned)
    sents = tokenize_sentences(cleaned)
    avg_w_len = float(np.mean([len(w) for w in words])) if words else 0.0

    return {
        "word_count": len(words),
        "character_count": chars,
        "sentence_count": max(1, len(sents)),
        "avg_word_length": round(avg_w_len, 2),
        "vocabulary_size": len(set(w.lower() for w in words))
    }


def filter_sentences_by_length(
    df: pd.DataFrame,
    min_words: int = 3,
    max_words: int = 100,
    col_name: str = "error_sentence"
) -> pd.DataFrame:
    """Filters out sentence pairs whose error sentence is outside length bounds."""
    if col_name not in df.columns:
        return df
    lengths = df[col_name].apply(lambda s: len(str(s).split()))
    return df[(lengths >= min_words) & (lengths <= max_words)].reset_index(drop=True)


def verify_data_integrity(df: pd.DataFrame) -> bool:
    """Validates presence of required columns, non-null values, and non-empty strings."""
    if df is None or df.empty:
        return False
    required = ["error_sentence", "corrected_sentence"]
    for r in required:
        if r not in df.columns:
            return False
        if df[r].isnull().any():
            return False
        if (df[r].astype(str).str.strip() == "").any():
            return False
    return True


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Applies full cleaning and punctuation normalization across dataframe."""
    clean_df = df.copy()
    if "error_sentence" in clean_df.columns:
        clean_df["error_sentence"] = clean_df["error_sentence"].apply(normalize_punctuation)
    if "corrected_sentence" in clean_df.columns:
        clean_df["corrected_sentence"] = clean_df["corrected_sentence"].apply(normalize_punctuation)
    return clean_df


class PreprocessingModule:
    """Object-oriented wrapper for preprocessing operations."""

    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.quote_pattern = re.compile(r'[\u2018\u2019\u201C\u201D]')

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def normalize_punctuation(self, text: str) -> str:
        return normalize_punctuation(text)

    def detect_heuristic_error_type(self, error_text: str, corr_text: str) -> str:
        """Heuristically infers error category by comparing word and character differences."""
        err_clean = clean_text(error_text).lower()
        corr_clean = clean_text(corr_text).lower()

        err_tokens = err_clean.split()
        corr_tokens = corr_clean.split()

        if len(err_tokens) == len(corr_tokens):
            has_verb_diff = any(
                e.endswith('s') != c.endswith('s') or e.endswith('ed') != c.endswith('ed')
                for e, c in zip(err_tokens, corr_tokens)
            )
            if has_verb_diff:
                return "Subject-Verb Agreement / Tense"
            return "Spelling & Word Form"

        if len(err_tokens) < len(corr_tokens):
            return "Missing Word / Article / Preposition"

        if len(err_tokens) > len(corr_tokens):
            return "Redundant Word / Conjunction"

        return "Syntax & Word Order"

    def batch_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = preprocess_corpus(df)
        if "error_type" not in clean_df.columns and "error_sentence" in clean_df.columns and "corrected_sentence" in clean_df.columns:
            clean_df["error_type"] = [
                self.detect_heuristic_error_type(e, c)
                for e, c in zip(clean_df["error_sentence"], clean_df["corrected_sentence"])
            ]
        return clean_df
