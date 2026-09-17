# -*- coding: utf-8 -*-
"""
data_preprocessing.py
---------------------
Text cleaning, normalization, sentence tokenization, length filtering,
and data integrity verification for text summarization documents.
"""

import re
import html
import unicodedata
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def clean_text(text: str) -> str:
    """
    Cleans raw document text by:
    - Normalizing Unicode characters (NFKC)
    - Decoding HTML entities (e.g. &nbsp;, &amp;)
    - Stripping HTML/XML tags
    - Standardizing quotation marks and dashes
    - Normalizing whitespace and newlines
    - Stripping leading/trailing blanks
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Replace non-breaking spaces and unusual whitespace
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    
    # Standardize curly quotes and dashes
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    
    # Remove space before punctuation if any
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Collapse multiple whitespace/newlines to single space
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def tokenize_sentences(text: str) -> List[str]:
    """
    Splits text into sentences cleanly using punctuation and boundary delimiters.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 5:
            sentences.append(s)
    return sentences if sentences else [cleaned]


def compute_text_statistics(text: str) -> Dict[str, Any]:
    """Compute word count, character count, sentence count, and average word length."""
    cleaned = clean_text(text)
    words = cleaned.split()
    sentences = tokenize_sentences(cleaned)
    char_count = len(cleaned)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_len = char_count / word_count if word_count > 0 else 0.0
    avg_sent_len = word_count / sentence_count if sentence_count > 0 else 0.0
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length_words": round(avg_sent_len, 2)
    }


def filter_documents_by_length(
    df: pd.DataFrame,
    min_words: int = 50,
    max_words: int = 3000,
    text_column: str = "document_text"
) -> pd.DataFrame:
    """Filter out documents that do not meet word count limits."""
    def is_valid(doc: str) -> bool:
        w_len = len(str(doc).split())
        return min_words <= w_len <= max_words
    
    filtered_df = df[df[text_column].apply(is_valid)].copy().reset_index(drop=True)
    return filtered_df


def preprocess_corpus(
    df: pd.DataFrame,
    min_words: int = 50,
    max_words: int = 3000
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline for the corpus DataFrame:
    - Cleans document_text and reference_summary
    - Computes length statistics
    - Filters by word limits
    - Validates integrity
    """
    processed = df.copy()
    processed["document_text"] = processed["document_text"].apply(clean_text)
    if "reference_summary" in processed.columns:
        processed["reference_summary"] = processed["reference_summary"].apply(clean_text)
    
    # Calculate stats
    processed["doc_word_count"] = processed["document_text"].apply(lambda t: len(t.split()))
    processed["doc_sentence_count"] = processed["document_text"].apply(lambda t: len(tokenize_sentences(t)))
    
    if "reference_summary" in processed.columns:
        processed["ref_word_count"] = processed["reference_summary"].apply(lambda t: len(t.split()))
    
    # Filter
    processed = filter_documents_by_length(processed, min_words=min_words, max_words=max_words)
    return processed


def verify_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify data integrity: missing values, empty strings, duplicate IDs."""
    missing_docs = int(df["document_text"].isna().sum())
    empty_docs = int((df["document_text"].str.strip() == "").sum())
    duplicate_ids = int(df["id"].duplicated().sum()) if "id" in df.columns else 0
    total_records = len(df)
    
    status = (missing_docs == 0) and (empty_docs == 0) and (duplicate_ids == 0) and (total_records > 0)
    
    return {
        "status": "PASSED" if status else "FAILED",
        "total_records": total_records,
        "missing_documents": missing_docs,
        "empty_documents": empty_docs,
        "duplicate_ids": duplicate_ids,
        "average_document_word_count": round(float(df["doc_word_count"].mean()), 2) if "doc_word_count" in df.columns else 0.0
    }
