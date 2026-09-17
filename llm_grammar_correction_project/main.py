#!/usr/bin/env python3
"""
Experiment 8: Automated Grammar Correction and Text Rewriting using Large Language Models (LLM)
Main Execution Script
Author: Mekesh Kumar M
"""

import os
import json
import pandas as pd
from typing import Dict, List

from src.data_loader import DataLoaderModule
from src.data_preprocessing import PreprocessingModule
from src.prompt_templates import PromptTemplates, PromptManager
from src.corrector import GrammarCorrector
from src.evaluation import EvaluationModule
from src.error_analyzer import ErrorAnalyzerModule
from src.visualization import VisualizationModule


def run_experiment_pipeline():
    print("\n" + "=" * 70)
    print("   EXPERIMENT 8: AUTOMATED GRAMMAR CORRECTION & TEXT REWRITING SYSTEM   ")
    print("      Multi-Tiered Prompt Templates & Sequence-to-Sequence LLMs      ")
    print("   Evaluation: Exact Match, Levenshtein Distance, Token F1, Over-Edit  ")
    print("=" * 70 + "\n")

    # [1/7] Config & Env
    print("[1/7] Loading configuration and initializing environment...")
    results_dir = "results"
    data_dir = "data"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "outputs"), exist_ok=True)

    config_path = os.path.join("config", "model_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    print(f"      [+] Config loaded. Active Mode: {config.get('active_mode', 'ollama')}")

    # [2/7] Data Acquisition & Preprocessing
    print("\n[2/7] Acquiring raw sentences and executing text preprocessing...")
    data_loader = DataLoaderModule()
    raw_df = data_loader.load_raw_data()
    stats = data_loader.verify_data_integrity(raw_df)
    print(f"      [+] Loaded raw corpus: {stats['total_pairs']} sentence pairs.")
    print(f"      [+] Data integrity check: PASSED (Avg error length: {stats['avg_error_words']:.1f} words).")

    preprocessor = PreprocessingModule()
    clean_df = preprocessor.batch_preprocess(raw_df)
    clean_path = os.path.join(data_dir, "processed", "cleaned_sentences.csv")
    data_loader.save_processed_data(clean_df, clean_path)

    train_df, val_df, test_df = data_loader.split_dataset(clean_df, 0.8, 0.1, 0.1)
    print(f"      [+] Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test.")

    # [3/7] Prompt Manager & Engine Initialization
    print("\n[3/7] Initializing Prompt Manager & Grammar Correction Engines...")
    prompt_manager = PromptManager(default_version="standard")
    print(f"      [+] Available prompt templates: {prompt_manager.list_available_versions()}")

    corrector = GrammarCorrector()
    evaluator = EvaluationModule()
    analyzer = ErrorAnalyzerModule()
    visualizer = VisualizationModule(output_dir=results_dir)

    # [4/7] Multi-Template Correction Execution
    print("\n[4/7] Generating corrections across multi-level prompt templates...")
    test_inputs = test_df["error_sentence"].tolist()
    test_refs = test_df["corrected_sentence"].tolist()
    test_types = test_df.get("error_type", ["Grammar"] * len(test_df)).tolist()

    prompt_comparison_results = {}
    for p_ver in ["minimal", "standard", "rewrite", "academic"]:
        preds = [corrector.correct_sentence(s, mode=p_ver) for s in test_inputs]
        summary, _ = evaluator.evaluate_batch(test_inputs, preds, test_refs, test_types)
        prompt_comparison_results[p_ver] = summary
        print(f"      [+] Tested '{p_ver:<12}' | Match: {summary['exact_match_accuracy']}% | Avg Lev Dist: {summary['avg_levenshtein_distance']} | Token F1: {summary['avg_token_f1']}")

    # Standard model run for primary evaluation
    std_preds = [corrector.correct_sentence(s, mode="standard") for s in test_inputs]
    eval_summary, eval_df = evaluator.evaluate_batch(test_inputs, std_preds, test_refs, test_types)

    # [5/7] Evaluation Metric Aggregation
    print("\n[5/7] Evaluating corrections (Exact Match %, Levenshtein Edit Distance, Token F1)...")
    print("-" * 75)
    print(f"{'ID':<3} | {'Lev Dist':<9} | {'Match':<6} | {'Token F1':<8} | {'Input Error Sentence':<35}")
    print("-" * 75)
    for _, row in eval_df.iterrows():
        print(f"{row['id']:<3} | {row['levenshtein_distance']:<9} | {str(row['exact_match']):<6} | {row['token_f1']:<8.4f} | {row['input_sentence'][:35]:<35}")
    print("-" * 75)

    print(f"\n[+] Overall Benchmark Summary (Standard Prompt):")
    print(f"    * Exact Match Accuracy     : {eval_summary['exact_match_accuracy']}%")
    print(f"    * Avg Levenshtein Distance : {eval_summary['avg_levenshtein_distance']} characters")
    print(f"    * Avg Token F1-Score       : {eval_summary['avg_token_f1']}")
    print(f"    * Over-Correction Rate     : {eval_summary['over_correction_rate']}%")
    print(f"    * Under-Correction Rate    : {eval_summary['under_correction_rate']}%")

    # [6/7] Error Category Analysis & Diagnostics
    print("\n[6/7] Performing error category breakdown and failure diagnostics...")
    by_type_df = analyzer.analyze_by_error_type(eval_df)
    if not by_type_df.empty:
        by_type_csv = os.path.join(results_dir, "performance_by_error_type.csv")
        by_type_df.to_csv(by_type_csv, index=False)
        print(f"      [+] Saved error type performance breakdown to {by_type_csv}")

    diagnostic_report = analyzer.generate_diagnostic_summary(eval_df)
    report_file = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(diagnostic_report)
    print(f"      [+] Saved comprehensive diagnostic report to {report_file}")

    # [7/7] Visualization & Artifact Rendering
    print("\n[7/7] Rendering evaluation plots and writing final reports...")
    p1 = visualizer.plot_prompt_comparison(prompt_comparison_results, "prompt_comparison.png")
    p2 = visualizer.plot_error_type_distribution(data_loader.categorize_errors(clean_df), "error_type_distribution.png")
    p3 = visualizer.plot_metric_distributions(eval_df, "metric_distributions.png")
    p4 = visualizer.plot_over_vs_under_correction(eval_df, "over_vs_under_correction.png")
    print(f"      [+] Generated plot: {p1}")
    print(f"      [+] Generated plot: {p2}")
    print(f"      [+] Generated plot: {p3}")
    print(f"      [+] Generated plot: {p4}")

    # Save detailed outputs
    scores_csv = os.path.join(results_dir, "accuracy_scores.csv")
    eval_df.to_csv(scores_csv, index=False)

    outputs_json = os.path.join(data_dir, "outputs", "corrected_sentences.json")
    with open(outputs_json, "w", encoding="utf-8") as f:
        json.dump(eval_df.to_dict(orient="records"), f, indent=2)

    sample_corrections_txt = os.path.join(results_dir, "sample_corrections.txt")
    with open(sample_corrections_txt, "w", encoding="utf-8") as f:
        f.write("EXPERIMENT 8: SAMPLE SENTENCE CORRECTIONS COMPARISON\n" + "=" * 65 + "\n\n")
        for _, row in eval_df.iterrows():
            f.write(f"Sample #{row['id']} [{row.get('error_type', 'General')}]:\n")
            f.write(f"  - Input Error : {row['input_sentence']}\n")
            f.write(f"  - Corrected   : {row['prediction']}\n")
            f.write(f"  - Gold Ref    : {row['reference']}\n")
            f.write(f"  - Lev Dist    : {row['levenshtein_distance']} chars | Match: {row['exact_match']}\n\n")

    print(f"      [+] Saved evaluation metrics to {scores_csv}")
    print(f"      [+] Saved test outputs to {outputs_json}")

    print("\n" + "=" * 70)
    print("   EXPERIMENT 8 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_experiment_pipeline()
