"""
PreprocessingModule for Experiment 8: Grammar Correction & Text Rewriting
Cleans text, standardizes whitespace and punctuation, and extracts error attributes.
"""

import re
import pandas as pd
from typing import List, Dict, Optional

class PreprocessingModule:
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.quote_pattern = re.compile(r'[\u2018\u2019\u201C\u201D]')

    def clean_text(self, text: str) -> str:
        """Standardizes encoding, quotes, and whitespace."""
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize smart quotes
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201C', '"').replace('\u201D', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '--')

        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        return text

    def normalize_punctuation(self, text: str) -> str:
        """Standardizes spacing around common punctuation marks."""
        text = self.clean_text(text)
        # Fix space before comma, period, question mark
        text = re.sub(r'\s+([,.:;?!])', r'\1', text)
        # Ensure single space after punctuation if followed by a letter
        text = re.sub(r'([,.:;?!])([A-Za-z])', r'\1 \2', text)
        return text

    def detect_heuristic_error_type(self, error_text: str, corr_text: str) -> str:
        """Heuristically infers error category by comparing word and character differences."""
        err_clean = self.clean_text(error_text).lower()
        corr_clean = self.clean_text(corr_text).lower()

        err_tokens = err_clean.split()
        corr_tokens = corr_clean.split()

        if len(err_tokens) == len(corr_tokens):
            # Check for singular/plural or tense differences
            has_verb_diff = any(e.endswith('s') != c.endswith('s') or e.endswith('ed') != c.endswith('ed') for e, c in zip(err_tokens, corr_tokens))
            if has_verb_diff:
                return "Subject-Verb Agreement / Tense"
            return "Spelling & Word Form"
        
        if len(err_tokens) < len(corr_tokens):
            return "Missing Word / Article / Preposition"
        
        if len(err_tokens) > len(corr_tokens):
            return "Redundant Word / Conjunction"
        
        return "Syntax & Word Order"

    def batch_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies cleaning and standardization to all sentence pairs in a DataFrame."""
        clean_df = df.copy()
        if "error_sentence" in clean_df.columns:
            clean_df["error_sentence"] = clean_df["error_sentence"].apply(self.normalize_punctuation)
        if "corrected_sentence" in clean_df.columns:
            clean_df["corrected_sentence"] = clean_df["corrected_sentence"].apply(self.normalize_punctuation)
        
        if "error_type" not in clean_df.columns and "error_sentence" in clean_df.columns and "corrected_sentence" in clean_df.columns:
            clean_df["error_type"] = [
                self.detect_heuristic_error_type(e, c) 
                for e, c in zip(clean_df["error_sentence"], clean_df["corrected_sentence"])
            ]
        
        return clean_df
