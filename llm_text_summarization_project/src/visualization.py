# -*- coding: utf-8 -*-
"""
visualization.py
----------------
Generates publication-quality charts and comparison plots for
Experiment 7: LLM Text Summarization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_prompt_comparison(
    prompt_df: pd.DataFrame,
    output_path: str = "results/prompt_comparison.png"
) -> None:
    """
    Plots a multi-metric grouped bar chart comparing performance across
    different prompt engineering strategies (v1, v2, best_prompt).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    prompt_labels = prompt_df["prompt_version"].tolist()
    x = np.arange(len(prompt_labels))
    width = 0.22

    r1 = prompt_df["rouge1_f1"].values
    r2 = prompt_df["rouge2_f1"].values
    rl = prompt_df["rougeL_f1"].values
    bs = prompt_df["bertscore_f1"].values if "bertscore_f1" in prompt_df.columns else None

    rects1 = ax.bar(x - 1.5 * width if bs is not None else x - width, r1, width, label="ROUGE-1 F1", color="#2b5c8f")
    rects2 = ax.bar(x - 0.5 * width if bs is not None else x, r2, width, label="ROUGE-2 F1", color="#e27c3e")
    rects3 = ax.bar(x + 0.5 * width if bs is not None else x + width, rl, width, label="ROUGE-L F1", color="#3b9a59")
    
    if bs is not None:
        rects4 = ax.bar(x + 1.5 * width, bs, width, label="BERTScore F1", color="#9357a7")

    ax.set_ylabel("Evaluation Score (F1)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Prompt Engineering Performance Comparison (Experiment 7)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_labels, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=10)
    ax.set_ylim(0, 1.05)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    if bs is not None:
        autolabel(rects4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved prompt comparison chart: {output_path}")


def plot_rouge_score_distributions(
    eval_df: pd.DataFrame,
    output_path: str = "results/metric_distributions.png"
) -> None:
    """
    Plots Precision, Recall, and F1 distribution across all ROUGE metrics and BERTScore.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    metrics = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]
    colors = ["#2b5c8f", "#e27c3e", "#3b9a59"]

    for idx, (m_key, m_title) in enumerate(metrics):
        ax = axes[idx]
        data = [
            eval_df[f"{m_key}_precision"],
            eval_df[f"{m_key}_recall"],
            eval_df[f"{m_key}_f1"]
        ]
        labels = ["Precision", "Recall", "F1-Score"]
        
        bplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.7)
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_title(f"{m_title} Score Distribution", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)

    plt.suptitle("Quantitative Evaluation Metric Distributions across Test Corpus", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved metric distributions chart: {output_path}")


def plot_length_and_compression(
    eval_df: pd.DataFrame,
    output_path: str = "results/length_and_compression.png"
) -> None:
    """
    Plots document word count vs summary word count along with compression efficiency.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Word Count Comparison
    indices = np.arange(len(eval_df))
    w = 0.35
    doc_words = eval_df["doc_word_count"] if "doc_word_count" in eval_df.columns else [150] * len(eval_df)
    sum_words = eval_df["summary_words"]

    ax1.bar(indices - w/2, doc_words, w, label="Original Document", color="#7f8c8d", alpha=0.85)
    ax1.bar(indices + w/2, sum_words, w, label="Generated Summary", color="#3498db", alpha=0.85)
    ax1.set_xlabel("Document Sample Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Word Count", fontsize=11, fontweight="bold")
    ax1.set_title("Original Document vs. Generated Summary Length", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white")

    # Subplot 2: Compression Ratio Distribution
    comp_pct = eval_df["word_compression_pct"] if "word_compression_pct" in eval_df.columns else [75.0] * len(eval_df)
    ax2.bar(indices, comp_pct, color="#2ecc71", alpha=0.85, edgecolor="#27ae60")
    ax2.axhline(np.mean(comp_pct), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean Compression: {np.mean(comp_pct):.1f}%")
    ax2.set_xlabel("Document Sample Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Compression Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Information Compression Efficiency", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved length and compression chart: {output_path}")


def plot_abstractive_vs_extractive(
    comparison_df: pd.DataFrame,
    output_path: str = "results/abstractive_vs_extractive.png"
) -> None:
    """
    Bar chart comparing Abstractive LLM (BART) vs Extractive Baseline (TextRank/TF-IDF).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    approaches = comparison_df["approach"].tolist()
    x = np.arange(len(approaches))
    width = 0.2

    r1 = comparison_df["rouge1_f1"].values
    r2 = comparison_df["rouge2_f1"].values
    rl = comparison_df["rougeL_f1"].values
    bs = comparison_df["bertscore_f1"].values

    ax.bar(x - 1.5 * width, r1, width, label="ROUGE-1", color="#1f77b4")
    ax.bar(x - 0.5 * width, r2, width, label="ROUGE-2", color="#ff7f0e")
    ax.bar(x + 0.5 * width, rl, width, label="ROUGE-L", color="#2ca02c")
    ax.bar(x + 1.5 * width, bs, width, label="BERTScore", color="#9467bd")

    ax.set_ylabel("Metric Score (F1)", fontsize=11, fontweight="bold")
    ax.set_title("Abstractive LLM vs. Extractive Baseline Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved abstractive vs extractive chart: {output_path}")
