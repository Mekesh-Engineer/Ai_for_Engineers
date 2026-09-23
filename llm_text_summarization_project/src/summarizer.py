# -*- coding: utf-8 -*-
"""
summarizer.py
-------------
Implements Abstractive LLM summarization (via Hugging Face AutoTokenizer and
AutoModelForSeq2SeqLM), local model persistence in models/saved_models/summarizer_model,
Extractive baseline summarization (TextRank / TF-IDF), and prompt engineering.
"""

import os
import re
import math
import time
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Suppress minor transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Hugging Face Transformers imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
    hf_logging.set_verbosity_error()
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class ExtractiveSummarizer:
    """
    Extractive Summarization baseline using TF-IDF and TextRank-style
    sentence salience scoring. Selects the most informative original sentences
    without generating new vocabulary.
    """

    def __init__(self, num_sentences: int = 3):
        self.num_sentences = num_sentences
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
            "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
            "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
            "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
            "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
            "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
            "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
            "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
            "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves"
        }

    def _split_sentences(self, text: str) -> List[str]:
        raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw_sents if len(s.strip()) > 10]

    def _tokenize_words(self, sentence: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', sentence.lower())
        return [w for w in words if w not in self.stop_words and len(w) > 2]

    def summarize(self, document_text: str, num_sentences: Optional[int] = None) -> str:
        """Extract top K sentences based on TF-IDF word frequency and position salience."""
        k = num_sentences if num_sentences is not None else self.num_sentences
        sentences = self._split_sentences(document_text)
        
        if not sentences:
            return document_text.strip()
        if len(sentences) <= k:
            return " ".join(sentences)

        tf: Dict[str, int] = {}
        for s in sentences:
            w_list = self._tokenize_words(s)
            for w in w_list:
                tf[w] = tf.get(w, 0) + 1

        max_tf = max(tf.values()) if tf else 1
        sentence_scores: List[Tuple[int, float, str]] = []
        total_sents = len(sentences)
        
        for idx, s in enumerate(sentences):
            w_list = self._tokenize_words(s)
            if not w_list:
                score = 0.0
            else:
                tf_score = sum(tf.get(w, 0) / max_tf for w in w_list) / len(w_list)
                position_weight = 1.0 + 0.5 * (1.0 - (idx / total_sents))
                score = tf_score * position_weight
            sentence_scores.append((idx, score, s))

        top_k = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:k]
        top_k_chronological = sorted(top_k, key=lambda x: x[0])
        return " ".join([item[2] for item in top_k_chronological])


class AbstractiveSummarizer:
    """
    Abstractive LLM Summarizer using Hugging Face AutoModelForSeq2SeqLM & AutoTokenizer.
    Prioritizes loading locally saved models from models/saved_models/summarizer_model.
    """

    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        device: str = "auto",
        torch_dtype: str = "float32",
        local_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.device_str = device
        self.local_dir = local_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "saved_models", "summarizer_model"
        )
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cpu")
        self._init_model()

    def _init_model(self) -> None:
        """Initialize Tokenizer and Model from local directory or HuggingFace."""
        if not _TRANSFORMERS_AVAILABLE:
            print("[!] Transformers not available. Abstractive summarizer running in mock mode.")
            return

        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device_str in ["cuda", "gpu"]:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # 1. Check if local saved model exists
        if os.path.isdir(self.local_dir) and (
            os.path.exists(os.path.join(self.local_dir, "config.json")) or
            os.path.exists(os.path.join(self.local_dir, "model.safetensors")) or
            os.path.exists(os.path.join(self.local_dir, "pytorch_model.bin"))
        ):
            try:
                print(f"[*] Loading locally persisted model from '{self.local_dir}' on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.local_dir)
                self.model.to(self.device)
                self.model.eval()
                print(f"[+] Successfully loaded local model from: {self.local_dir}")
                return
            except Exception as e:
                print(f"[!] Could not load from local path: {e}. Falling back to hub...")

        # 2. Load from HuggingFace and save locally into models/saved_models/summarizer_model/
        try:
            print(f"[*] Loading base model '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"[+] Loaded model '{self.model_name}'. Persisting to '{self.local_dir}'...")
            self.save_local_checkpoint(self.local_dir)
        except Exception as e:
            print(f"[!] Failed to load '{self.model_name}': {e}")
            fallback = "t5-small"
            print(f"[*] Attempting fallback to '{fallback}'...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback)
                self.model.to(self.device)
                self.model.eval()
                self.model_name = fallback
                self.save_local_checkpoint(self.local_dir)
                print(f"[+] Fallback to '{fallback}' succeeded and saved locally.")
            except Exception as e2:
                print(f"[!] Fallback failed: {e2}. Summarizer will use extractive fallback.")
                self.tokenizer = None
                self.model = None

    def is_ready(self) -> bool:
        """Returns True if tokenizer and model are loaded and ready."""
        return self.model is not None and self.tokenizer is not None

    def save_local_checkpoint(self, target_dir: str) -> None:
        """Persist current model weights, tokenizer, and config to disk."""
        try:
            os.makedirs(target_dir, exist_ok=True)
            if self.model is not None and hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(target_dir)
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(target_dir)
            
            # Also save PyTorch checkpoint in models/
            models_root = os.path.dirname(os.path.dirname(os.path.abspath(target_dir)))
            pt_path = os.path.join(models_root, "summarizer_checkpoint.pt")
            if self.model is not None and hasattr(self.model, "state_dict"):
                torch.save(self.model.state_dict(), pt_path)
            print(f"[+] Model checkpoint saved to {target_dir}")
        except Exception as e:
            print(f"[!] Could not save local checkpoint: {e}")

    def summarize(
        self,
        text: str,
        min_length: int = 30,
        max_length: int = 120,
        num_beams: int = 2,
        temperature: float = 0.7,
        top_p: float = 0.9,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate abstractive summary for given text using Seq2Seq generation.
        """
        text = text.strip()
        if not text:
            return ""

        if self.model is None or self.tokenizer is None:
            extractive = ExtractiveSummarizer(num_sentences=2)
            return extractive.summarize(text)

        words = text.split()
        if len(words) < 20:
            return text

        if len(words) > 700:
            chunks = self._chunk_text(text, max_words=450)
            chunk_summaries = []
            for chunk in chunks:
                chunk_summaries.append(self._generate_single_pass(
                    chunk,
                    min_length=min_length // 2,
                    max_length=max(min_length + 10, max_length // len(chunks)),
                    num_beams=max(1, num_beams),
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                ))
            combined_text = " ".join(chunk_summaries)
            return self._generate_single_pass(
                combined_text,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )

        return self._generate_single_pass(
            text,
            min_length=min_length,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

    def _generate_single_pass(
        self,
        text: str,
        min_length: int,
        max_length: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> str:
        try:
            input_text = f"summarize: {text}" if "t5" in str(self.model_name).lower() else text
            
            inputs = self.tokenizer(
                input_text,
                max_length=1024,
                truncation=True,
                padding=False,
                return_tensors="pt"
            ).to(self.device)

            gen_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "early_stopping": kwargs.get("early_stopping", True)
            }
            if "bart" in str(self.model_name).lower():
                gen_kwargs["forced_bos_token_id"] = 0

            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            for k, v in kwargs.items():
                if k not in gen_kwargs and k not in ["device", "model_name"]:
                    gen_kwargs[k] = v

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs
                )

            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            summary = re.sub(r"\s+", " ", summary)
            return summary
        except Exception as e:
            print(f"[!] Generation error: {e}. Using extractive fallback.")
            return ExtractiveSummarizer(num_sentences=2).summarize(text)

    def _chunk_text(self, text: str, max_words: int = 450) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for s in sentences:
            s_words = len(s.split())
            if current_word_count + s_words > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_word_count = s_words
            else:
                current_chunk.append(s)
                current_word_count += s_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


def summarize_with_prompt_engineering(
    document_text: str,
    prompt_template_key: str = "best",
    summarizer: Optional[AbstractiveSummarizer] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Applies prompt engineering specifications to guide summarization parameters,
    executes model inference, and returns summary with operational metrics.
    """
    if summarizer is None:
        summarizer = AbstractiveSummarizer()

    kwargs = {
        "min_length": 30,
        "max_length": 120,
        "num_beams": 2,
        "length_penalty": 2.0,
        "no_repeat_ngram_size": 3,
        "do_sample": False,
        "early_stopping": True
    }
    if generation_kwargs:
        kwargs.update(generation_kwargs)

    # Adjust generation kwargs according to prompt template style
    if prompt_template_key == "v1":
        kwargs["max_length"] = 80
        kwargs["min_length"] = 30
        kwargs["num_beams"] = 2
    elif prompt_template_key == "v2":
        kwargs["max_length"] = 100
        kwargs["min_length"] = 35
        kwargs["num_beams"] = 2
    elif prompt_template_key == "best":
        kwargs["max_length"] = 120
        kwargs["min_length"] = 40
        kwargs["num_beams"] = 2
        kwargs["length_penalty"] = 2.0

    t0 = time.time()
    generated_text = summarizer.summarize(document_text, **kwargs)
    latency_sec = round(time.time() - t0, 3)

    return {
        "prompt_version": prompt_template_key,
        "summary_text": generated_text,
        "latency_seconds": latency_sec,
        "summary_word_count": len(generated_text.split()),
        "generation_parameters": kwargs
    }
