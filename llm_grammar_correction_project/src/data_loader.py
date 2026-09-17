"""
DataLoaderModule for Experiment 8: Automated Grammar Correction & Text Rewriting
Loads, validates, categorizes, and splits sentence error-correction corpora.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional

class DataLoaderModule:
    def __init__(self, raw_data_path: Optional[str] = None):
        self.raw_data_path = raw_data_path or os.path.join("data", "raw", "lang8_errors.csv")
        self.default_corpus = [
            {
                "id": 1,
                "error_sentence": "He go to the laboratory yesterday for doing the experiment.",
                "corrected_sentence": "He went to the laboratory yesterday to do the experiment.",
                "error_type": "Verb Tense & Preposition",
                "difficulty": "Easy",
                "source": "Lang-8"
            },
            {
                "id": 2,
                "error_sentence": "The datas collected by sensors was showing many error.",
                "corrected_sentence": "The data collected by sensors showed many errors.",
                "error_type": "Pluralization & Subject-Verb Agreement",
                "difficulty": "Medium",
                "source": "Lang-8"
            },
            {
                "id": 3,
                "error_sentence": "Neural network are very fast but it require GPU for speed up.",
                "corrected_sentence": "Neural networks are very fast, but they require GPUs for speedup.",
                "error_type": "Agreement & Punctuation",
                "difficulty": "Medium",
                "source": "Lang-8"
            },
            {
                "id": 4,
                "error_sentence": "I am look forward to collaborate with your engineering team.",
                "corrected_sentence": "I look forward to collaborating with your engineering team.",
                "error_type": "Verb Form & Preposition",
                "difficulty": "Easy",
                "source": "Lang-8"
            },
            {
                "id": 5,
                "error_sentence": "This algorithm is more superior than traditional methods.",
                "corrected_sentence": "This algorithm is superior to traditional methods.",
                "error_type": "Comparative Adjective & Preposition",
                "difficulty": "Medium",
                "source": "Lang-8"
            },
            {
                "id": 6,
                "error_sentence": "There is many different solution for solve this optimization problem.",
                "corrected_sentence": "There are many different solutions to solve this optimization problem.",
                "error_type": "Subject-Verb Agreement & Preposition",
                "difficulty": "Medium",
                "source": "Lang-8"
            },
            {
                "id": 7,
                "error_sentence": "Although it was raining heavy, but we continued the field test.",
                "corrected_sentence": "Although it was raining heavily, we continued the field test.",
                "error_type": "Conjunction Redundancy & Adverb Form",
                "difficulty": "Hard",
                "source": "Lang-8"
            },
            {
                "id": 8,
                "error_sentence": "The researcher discuss about the convergence rate of gradient descent.",
                "corrected_sentence": "The researcher discusses the convergence rate of gradient descent.",
                "error_type": "Preposition Redundancy & Verb Agreement",
                "difficulty": "Easy",
                "source": "Lang-8"
            },
            {
                "id": 9,
                "error_sentence": "We must to consider the thermal dissipation in embedded circuit.",
                "corrected_sentence": "We must consider thermal dissipation in the embedded circuit.",
                "error_type": "Modal Verb & Article",
                "difficulty": "Medium",
                "source": "Lang-8"
            },
            {
                "id": 10,
                "error_sentence": "Each of the component have been tested under high pressure.",
                "corrected_sentence": "Each of the components has been tested under high pressure.",
                "error_type": "Subject-Verb Agreement & Pluralization",
                "difficulty": "Hard",
                "source": "Lang-8"
            },
            {
                "id": 11,
                "error_sentence": "The model perform poorly because of it lacks of enough training data.",
                "corrected_sentence": "The model performs poorly because it lacks sufficient training data.",
                "error_type": "Clause Structure & Word Choice",
                "difficulty": "Hard",
                "source": "Lang-8"
            },
            {
                "id": 12,
                "error_sentence": "Its important to verify that all variable is initialized properly.",
                "corrected_sentence": "It is important to verify that all variables are initialized properly.",
                "error_type": "Contraction & Subject-Verb Agreement",
                "difficulty": "Easy",
                "source": "Lang-8"
            },
            {
                "id": 13,
                "error_sentence": "We have received your informations regarding the firmware update.",
                "corrected_sentence": "We have received your information regarding the firmware update.",
                "error_type": "Uncountable Noun",
                "difficulty": "Easy",
                "source": "Lang-8"
            },
            {
                "id": 14,
                "error_sentence": "The results demonstrates that our proposed architecture is effecient.",
                "corrected_sentence": "The results demonstrate that our proposed architecture is efficient.",
                "error_type": "Subject-Verb Agreement & Spelling",
                "difficulty": "Easy",
                "source": "Lang-8"
            },
            {
                "id": 15,
                "error_sentence": "If the voltage will increase, the semiconductor device can damaged.",
                "corrected_sentence": "If the voltage increases, the semiconductor device can be damaged.",
                "error_type": "Conditional Tense & Passive Voice",
                "difficulty": "Hard",
                "source": "Lang-8"
            }
        ]

    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Loads dataset from CSV or returns the built-in gold standard benchmark."""
        path = filepath or self.raw_data_path
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Standardize column naming if necessary
                col_map = {
                    "input": "error_sentence",
                    "input_sentence": "error_sentence",
                    "reference": "corrected_sentence",
                    "reference_correction": "corrected_sentence",
                    "error_category": "error_type"
                }
                df = df.rename(columns=col_map)
                if "error_sentence" in df.columns and "corrected_sentence" in df.columns:
                    return df
            except Exception as e:
                print(f"[!] Warning: Could not read CSV ({e}). Using default corpus.")

        df = pd.DataFrame(self.default_corpus)
        return df

    def verify_data_integrity(self, df: pd.DataFrame) -> Dict:
        """Validates alignment, non-empty pairs, and computes dataset length statistics."""
        if df.empty:
            raise ValueError("Dataset is empty.")
        
        required_cols = ["error_sentence", "corrected_sentence"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in dataset: {col}")

        # Clean nulls
        clean_df = df.dropna(subset=required_cols).copy()
        clean_df["error_sentence"] = clean_df["error_sentence"].astype(str).str.strip()
        clean_df["corrected_sentence"] = clean_df["corrected_sentence"].astype(str).str.strip()
        clean_df = clean_df[(clean_df["error_sentence"] != "") & (clean_df["corrected_sentence"] != "")]

        error_lens = clean_df["error_sentence"].apply(lambda s: len(s.split()))
        corr_lens = clean_df["corrected_sentence"].apply(lambda s: len(s.split()))

        stats = {
            "total_pairs": len(clean_df),
            "avg_error_words": float(error_lens.mean()),
            "avg_corr_words": float(corr_lens.mean()),
            "min_error_words": int(error_lens.min()),
            "max_error_words": int(error_lens.max()),
            "integrity_passed": len(clean_df) > 0
        }
        return stats

    def categorize_errors(self, df: pd.DataFrame) -> Dict[str, int]:
        """Classifies and counts distribution across error categories."""
        if "error_type" not in df.columns:
            return {"General Grammar": len(df)}
        counts = df["error_type"].value_counts().to_dict()
        return counts

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the sentence pairs into train, validation, and test subsets."""
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(shuffled)
        if n == 0:
            return shuffled.copy(), shuffled.copy(), shuffled.copy()

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_val == 0 and n >= 3:
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0 and n >= 3:
            n_test = 1
            n_train = n - n_val - n_test

        train_df = shuffled.iloc[:n_train].reset_index(drop=True)
        val_df = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = shuffled.iloc[n_train + n_val:].reset_index(drop=True)

        return train_df, val_df, test_df

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Saves processed sentences to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
