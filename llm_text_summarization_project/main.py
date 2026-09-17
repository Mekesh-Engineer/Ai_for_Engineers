# -*- coding: utf-8 -*-
"""
main.py
-------
End-to-end execution pipeline for Experiment 7: Text Summarization using
Large Language Models (LLMs), Prompt Engineering, and Local Checkpointing.

Usage:
    py main.py
    py main.py --train
    py main.py --model sshleifer/distilbart-cnn-12-6 --prompt_version best
    py main.py --config config/model_config.json
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

# Suppress minor warnings
warnings.filterwarnings("ignore")

# Add src to python path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader import (
    load_config,
    get_or_create_raw_articles,
    split_dataset,
    save_processed_data,
    load_test_documents,
    load_reference_summaries,
    save_generated_summaries
)
from data_preprocessing import (
    preprocess_corpus,
    verify_data_integrity,
    compute_text_statistics
)
from prompt_templates import (
    PromptManager,
    create_summarization_prompt
)
from summarizer import (
    AbstractiveSummarizer,
    ExtractiveSummarizer,
    summarize_with_prompt_engineering
)
from model_trainer import (
    train_or_finetune_summarizer,
    save_model_artifacts
)
from evaluation import (
    calculate_rouge_scores,
    calculate_bertscore,
    assess_length_compliance,
    calculate_compression_ratio,
    evaluate_summary_quality,
    generate_evaluation_report
)
from visualization import (
    plot_prompt_comparison,
    plot_rouge_score_distributions,
    plot_length_and_compression,
    plot_abstractive_vs_extractive
)


def print_banner() -> None:
    banner = (
        "\n"
        "  +==============================================================+\n"
        "  |        EXPERIMENT 7: LLM-BASED TEXT SUMMARIZATION SYSTEM     |\n"
        "  |   Abstractive vs. Extractive Summarization & Prompt Tuning   |\n"
        "  |    Evaluation: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, Ratio   |\n"
        "  +==============================================================+\n"
    )
    print(banner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM Text Summarization Pipeline — Experiment 7"
    )
    default_cfg = os.path.join(_PROJECT_ROOT, "config", "model_config.json")
    parser.add_argument("--config", default=default_cfg, help="Path to model_config.json")
    parser.add_argument("--model", default=None, help="Hugging Face model name override")
    parser.add_argument("--prompt_version", default="best", choices=["v1", "v2", "best"], help="Prompt engineering template")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--train", action="store_true", help="Fine-tune model on train set and save to models/saved_models/")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs if --train is active")
    return parser.parse_args()


def main() -> None:
    t_start = time.time()
    args = parse_args()
    print_banner()

    # ── 1. Configuration & Reproducibility ──────────────────────────────────
    print("[1/7] Loading configuration and initializing environment...")
    cfg = load_config(args.config)
    seed = cfg.get("execution", {}).get("random_seed", 42)
    np.random.seed(seed)

    model_name = args.model if args.model else cfg["model"]["default_model_name"]
    prompt_version = args.prompt_version
    device = args.device if args.device != "auto" else cfg["model"].get("device", "auto")
    local_model_dir = os.path.join(_PROJECT_ROOT, "models", "saved_models", "summarizer_model")

    # ── 2. Data Acquisition & Preprocessing ─────────────────────────────────
    print("\n[2/7] Acquiring raw articles and executing text preprocessing...")
    raw_csv = os.path.join(_PROJECT_ROOT, cfg["dataset"]["raw_csv_path"])
    raw_df = get_or_create_raw_articles(raw_csv)
    print(f"      [+] Loaded raw corpus: {len(raw_df)} documents.")

    processed_df = preprocess_corpus(
        raw_df,
        min_words=cfg["dataset"].get("min_doc_words", 50),
        max_words=cfg["dataset"].get("max_doc_words", 3000)
    )
    integrity = verify_data_integrity(processed_df)
    print(f"      [+] Data integrity check: {integrity['status']} (Avg doc length: {integrity['average_document_word_count']} words).")

    # Split dataset into train/val/test (6 train, 1 validation, 3 test)
    train_df, val_df, test_df = split_dataset(
        processed_df,
        train_ratio=cfg["dataset"].get("train_ratio", 0.8),
        val_ratio=cfg["dataset"].get("val_ratio", 0.1),
        test_ratio=cfg["dataset"].get("test_ratio", 0.1),
        seed=seed
    )
    print(f"      [+] Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test.")

    # Save data artifacts
    save_processed_data(
        processed_df,
        test_df,
        cleaned_csv_path=os.path.join(_PROJECT_ROOT, cfg["dataset"]["cleaned_csv_path"]),
        test_docs_json_path=os.path.join(_PROJECT_ROOT, cfg["dataset"]["test_docs_json_path"]),
        reference_summaries_json_path=os.path.join(_PROJECT_ROOT, cfg["dataset"]["reference_summaries_json_path"])
    )

    # ── Optional: Fine-tune & Save Model Checkpoint ──────────────────────────
    if args.train:
        print("\n[*] Fine-tuning Seq2Seq Summarizer on training split...")
        train_or_finetune_summarizer(
            train_df,
            val_df,
            base_model_name=model_name,
            output_dir=local_model_dir,
            epochs=args.epochs,
            batch_size=cfg.get("execution", {}).get("batch_size", 2),
            device=device
        )

    # ── 3. Initialize Prompt Manager & Models ───────────────────────────────
    print("\n[3/7] Initializing Prompt Manager & Summarization Engines...")
    prompt_dir = os.path.join(_PROJECT_ROOT, "models", "prompts")
    prompt_mgr = PromptManager(prompt_dir=prompt_dir)
    print(f"      [+] Available prompt templates: {prompt_mgr.get_prompt_keys()} (Active: '{prompt_version}')")

    abstractive_model = AbstractiveSummarizer(
        model_name=model_name,
        device=device,
        local_dir=local_model_dir
    )
    extractive_model = ExtractiveSummarizer(num_sentences=cfg["extractive"].get("num_sentences", 3))

    # ── 4. Generate Summaries on Test Set ───────────────────────────────────
    print(f"\n[4/7] Generating summaries on test documents using Prompt '{prompt_version}'...")
    test_records = test_df.to_dict(orient="records")
    results = []
    generated_summaries_json = []

    for idx, doc in enumerate(test_records, 1):
        doc_id = doc["id"]
        title = doc["title"]
        text = doc["document_text"]
        ref_sum = doc["reference_summary"]

        print(f"      [{idx}/{len(test_records)}] Processing '{doc_id}': {title[:35]}...")

        # Abstractive LLM Generation
        abs_res = summarize_with_prompt_engineering(
            document_text=text,
            prompt_template_key=prompt_version,
            summarizer=abstractive_model,
            generation_kwargs=cfg.get("generation", {})
        )
        gen_text = abs_res["summary_text"]
        latency = abs_res["latency_seconds"]

        # Extractive baseline generation
        ext_text = extractive_model.summarize(text)

        # Quality evaluation
        eval_metrics = evaluate_summary_quality(
            reference=ref_sum,
            hypothesis=gen_text,
            document=text,
            target_length_words=cfg["evaluation"].get("target_summary_length_words", 60),
            compute_bert=True
        )

        record_entry = {
            "id": doc_id,
            "category": doc.get("category", "General"),
            "title": title,
            "doc_word_count": len(text.split()),
            "ref_word_count": len(ref_sum.split()),
            "generated_summary": gen_text,
            "reference_summary": ref_sum,
            "extractive_summary": ext_text,
            "latency_seconds": latency,
            **eval_metrics
        }
        results.append(record_entry)
        generated_summaries_json.append({
            "id": doc_id,
            "title": title,
            "generated_summary": gen_text,
            "latency_seconds": latency,
            "metrics": eval_metrics
        })

    eval_df = pd.DataFrame(results)

    # ── 5. Prompt Engineering Comparison Experiment ────────────────────────
    print("\n[5/7] Executing Prompt Engineering Multi-Variation Comparison...")
    prompt_comp_records = []
    for p_ver in ["v1", "v2", "best"]:
        p_r1, p_r2, p_rl, p_bs, p_lat = [], [], [], [], []
        p_refs, p_hyps = [], []
        for doc in test_records:
            t_res = summarize_with_prompt_engineering(
                doc["document_text"],
                prompt_template_key=p_ver,
                summarizer=abstractive_model
            )
            r_scores = calculate_rouge_scores(doc["reference_summary"], t_res["summary_text"])
            p_r1.append(r_scores["rouge1"]["f1"])
            p_r2.append(r_scores["rouge2"]["f1"])
            p_rl.append(r_scores["rougeL"]["f1"])
            p_lat.append(t_res["latency_seconds"])
            p_refs.append(doc["reference_summary"])
            p_hyps.append(t_res["summary_text"])
        
        # Batch BERTScore computation
        b_score = calculate_bertscore(p_refs, p_hyps)
        prompt_comp_records.append({
            "prompt_version": f"Prompt {p_ver}",
            "rouge1_f1": round(float(np.mean(p_r1)), 4),
            "rouge2_f1": round(float(np.mean(p_r2)), 4),
            "rougeL_f1": round(float(np.mean(p_rl)), 4),
            "bertscore_f1": round(float(b_score["f1"]), 4),
            "latency_seconds": round(float(np.mean(p_lat)), 3)
        })
    prompt_comp_df = pd.DataFrame(prompt_comp_records)

    # ── 6. Abstractive vs Extractive Comparison ────────────────────────────
    print("\n[6/7] Evaluating Abstractive LLM vs. Extractive Baseline...")
    ext_r1, ext_r2, ext_rl = [], [], []
    ext_refs, ext_hyps = [], []
    for doc in test_records:
        ext_sum = extractive_model.summarize(doc["document_text"])
        eq = calculate_rouge_scores(doc["reference_summary"], ext_sum)
        ext_r1.append(eq["rouge1"]["f1"])
        ext_r2.append(eq["rouge2"]["f1"])
        ext_rl.append(eq["rougeL"]["f1"])
        ext_refs.append(doc["reference_summary"])
        ext_hyps.append(ext_sum)

    ext_b_score = calculate_bertscore(ext_refs, ext_hyps)

    abs_ext_df = pd.DataFrame([
        {
            "approach": "Abstractive (BART/LLM)",
            "rouge1_f1": round(float(eval_df["rouge1_f1"].mean()), 4),
            "rouge2_f1": round(float(eval_df["rouge2_f1"].mean()), 4),
            "rougeL_f1": round(float(eval_df["rougeL_f1"].mean()), 4),
            "bertscore_f1": round(float(eval_df["bertscore_f1"].mean()), 4)
        },
        {
            "approach": "Extractive (TextRank)",
            "rouge1_f1": round(float(np.mean(ext_r1)), 4),
            "rouge2_f1": round(float(np.mean(ext_r2)), 4),
            "rougeL_f1": round(float(np.mean(ext_rl)), 4),
            "bertscore_f1": round(float(ext_b_score["f1"]), 4)
        }
    ])

    # ── 7. Export Results, Visualizations & Reports ─────────────────────────
    print("\n[7/7] Exporting results, metrics, visualizations, and summary reports...")
    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Summaries output CSV
    summaries_csv_path = os.path.join(results_dir, "summaries_output.csv")
    eval_df.to_csv(summaries_csv_path, index=False, encoding="utf-8")
    print(f"      [+] Saved summaries output: {summaries_csv_path}")

    # 2. ROUGE scores CSV
    rouge_csv_path = os.path.join(results_dir, "rouge_scores.csv")
    eval_df[["id", "rouge1_precision", "rouge1_recall", "rouge1_f1",
             "rouge2_precision", "rouge2_recall", "rouge2_f1",
             "rougeL_precision", "rougeL_recall", "rougeL_f1",
             "bertscore_f1", "word_compression_pct"]].to_csv(rouge_csv_path, index=False, encoding="utf-8")
    print(f"      [+] Saved ROUGE scores CSV: {rouge_csv_path}")

    # 3. JSON generated summaries
    gen_sum_json_path = os.path.join(_PROJECT_ROOT, cfg["dataset"]["generated_summaries_json_path"])
    save_generated_summaries(generated_summaries_json, gen_sum_json_path)
    print(f"      [+] Saved generated summaries JSON: {gen_sum_json_path}")

    # 4. Text reports
    report_text = generate_evaluation_report(eval_df, prompt_comp_df, abs_ext_df)
    report_file_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"      [+] Saved evaluation report: {report_file_path}")

    # Sample Quality Assessment Report
    quality_file_path = os.path.join(results_dir, "quality_assessment.txt")
    with open(quality_file_path, "w", encoding="utf-8") as f:
        f.write("EXPERIMENT 7: SAMPLE SUMMARIES QUALITY ASSESSMENT\n")
        f.write("=" * 60 + "\n\n")
        for idx, row in eval_df.iterrows():
            f.write(f"[{row['id']}] {row['title']}\n")
            f.write(f"Category: {row['category']}\n")
            f.write(f"Original Length: {row['doc_word_count']} words | Compression: {row['word_compression_pct']}%\n\n")
            f.write("--- GROUND TRUTH REFERENCE SUMMARY ---\n")
            f.write(row["reference_summary"].strip() + "\n\n")
            f.write("--- GENERATED ABSTRACTIVE SUMMARY (LLM) ---\n")
            f.write(row["generated_summary"].strip() + "\n\n")
            f.write("--- EXTRACTIVE BASELINE SUMMARY ---\n")
            f.write(row["extractive_summary"].strip() + "\n\n")
            f.write(f"ROUGE-1 F1: {row['rouge1_f1']:.4f} | ROUGE-2 F1: {row['rouge2_f1']:.4f} | ROUGE-L F1: {row['rougeL_f1']:.4f} | BERTScore: {row['bertscore_f1']:.4f}\n")
            f.write("=" * 60 + "\n\n")
    print(f"      [+] Saved quality assessment report: {quality_file_path}")

    # 5. Visualizations
    plot_prompt_comparison(prompt_comp_df, output_path=os.path.join(results_dir, "prompt_comparison.png"))
    plot_rouge_score_distributions(eval_df, output_path=os.path.join(results_dir, "metric_distributions.png"))
    plot_length_and_compression(eval_df, output_path=os.path.join(results_dir, "length_and_compression.png"))
    plot_abstractive_vs_extractive(abs_ext_df, output_path=os.path.join(results_dir, "abstractive_vs_extractive.png"))

    elapsed = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"EXPERIMENT 7 COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS.")
    print(f"Mean ROUGE-1 F1: {eval_df['rouge1_f1'].mean():.4f} | Mean ROUGE-L F1: {eval_df['rougeL_f1'].mean():.4f} | Mean BERTScore: {eval_df['bertscore_f1'].mean():.4f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
