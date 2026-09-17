# -*- coding: utf-8 -*-
"""
model_trainer.py
----------------
Handles fine-tuning, training loops, checkpointing, and local model serialization
for Experiment 7: Text Summarization using Large Language Models (LLMs).
Saves trained model and tokenizer into models/saved_models/summarizer_model/.
"""

import os
import sys
import time
import json
import torch
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AdamW,
        get_linear_schedule_with_warmup
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def save_model_artifacts(
    model: Any,
    tokenizer: Any,
    output_dir: str = "models/saved_models/summarizer_model",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves model weights, configuration, tokenizer vocab/merges, and training metadata
    directly into the project models/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Saving model and tokenizer artifacts to: {output_dir} ...")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    # Save PyTorch checkpoint state dict file directly in models/
    models_root = os.path.dirname(os.path.dirname(os.path.abspath(output_dir)))
    pt_checkpoint_path = os.path.join(models_root, "summarizer_checkpoint.pt")
    if hasattr(model, "state_dict"):
        torch.save({
            "model_state_dict": model.state_dict(),
            "timestamp": time.time(),
            "model_type": getattr(model.config, "model_type", "bart") if hasattr(model, "config") else "seq2seq"
        }, pt_checkpoint_path)
        print(f"[+] Saved PyTorch checkpoint: {pt_checkpoint_path}")

    # Save metadata
    meta_path = os.path.join(output_dir, "training_metadata.json")
    meta_dict = metadata or {}
    meta_dict["saved_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"[+] Model artifacts successfully persisted to {output_dir}")


def train_or_finetune_summarizer(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    base_model_name: str = "sshleifer/distilbart-cnn-12-6",
    output_dir: str = "models/saved_models/summarizer_model",
    epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_input_length: int = 512,
    max_target_length: int = 128,
    device: str = "auto"
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Fine-tunes or trains the Seq2Seq summarization model on domain document-summary pairs
    and saves the model checkpoints locally into models/saved_models/summarizer_model/.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library is required for model training.")

    # Device selection
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device in ["cuda", "gpu"]:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    print("=" * 70)
    print(f"FINE-TUNING SUMMARIZATION MODEL: {base_model_name} on {dev}")
    print(f"Training samples: {len(train_df)} | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    print(f"[*] Initializing base architecture from '{base_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model.to(dev)

    # 2. Prepare Data
    def prepare_batch(df_slice: pd.DataFrame) -> Dict[str, torch.Tensor]:
        prefix = "summarize: " if "t5" in base_model_name.lower() else ""
        inputs = [prefix + str(doc) for doc in df_slice["document_text"].tolist()]
        targets = [str(ref) for ref in df_slice["reference_summary"].tolist()]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        # Replace padding token id with -100 for loss calculation
        label_ids = labels["input_ids"]
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids
        return {k: v.to(dev) for k, v in model_inputs.items()}

    # 3. Optimizer & Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history = {"train_loss": [], "epoch_times": []}

    model.train()
    n_samples = len(train_df)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        running_loss = 0.0
        shuffled_df = train_df.sample(frac=1.0, random_state=42 + epoch).reset_index(drop=True)

        for b_idx in range(n_batches):
            batch_slice = shuffled_df.iloc[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch_data = prepare_batch(batch_slice)

            optimizer.zero_grad()
            outputs = model(**batch_data)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(batch_slice)

        epoch_loss = running_loss / n_samples
        ep_duration = time.time() - t_ep_start
        history["train_loss"].append(round(epoch_loss, 4))
        history["epoch_times"].append(round(ep_duration, 2))
        print(f"[*] Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} - Time: {ep_duration:.2f}s")

    # 4. Save Fine-Tuned Model into models/saved_models/summarizer_model/
    metadata = {
        "base_model": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "final_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "training_history": history,
        "device_trained_on": str(dev)
    }
    save_model_artifacts(model, tokenizer, output_dir=output_dir, metadata=metadata)

    return model, tokenizer, history


if __name__ == "__main__":
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    from data_loader import get_or_create_raw_articles, split_dataset
    from data_preprocessing import preprocess_corpus

    raw_csv = os.path.join(_PROJECT_ROOT, "data", "raw", "articles.csv")
    df = preprocess_corpus(get_or_create_raw_articles(raw_csv))
    train, val, test = split_dataset(df)

    out_dir = os.path.join(_PROJECT_ROOT, "models", "saved_models", "summarizer_model")
    train_or_finetune_summarizer(train, val, output_dir=out_dir, epochs=1, batch_size=2)
