"""
VisualizationModule for Experiment 8: Automated Grammar Correction & Text Rewriting
Generates publication-quality charts for prompt comparisons, metric distributions, and error analyses.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional

class VisualizationModule:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Professional styling
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10

    def plot_prompt_comparison(self, comparison_data: Dict[str, Dict], filename: str = "prompt_comparison.png") -> str:
        """Plots Exact Match Accuracy, Token F1, and Levenshtein Distance across prompt templates."""
        out_path = os.path.join(self.output_dir, filename)
        
        prompts = list(comparison_data.keys())
        exact_matches = [comparison_data[p].get("exact_match_accuracy", 0) for p in prompts]
        f1_scores = [comparison_data[p].get("avg_token_f1", 0) * 100 for p in prompts]
        lev_dists = [comparison_data[p].get("avg_levenshtein_distance", 0) for p in prompts]

        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
        width = 0.28
        x = range(len(prompts))

        rects1 = ax1.bar([i - width/2 for i in x], exact_matches, width, label='Exact Match Accuracy (%)', color='#6366f1', alpha=0.9)
        rects2 = ax1.bar([i + width/2 for i in x], f1_scores, width, label='Token F1 Score (×100)', color='#10b981', alpha=0.9)

        ax1.set_ylabel('Performance Score (%)', fontweight='bold')
        ax1.set_xlabel('Prompt Template Version', fontweight='bold')
        ax1.set_title('Experiment 8: Prompt Template Performance Comparison', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels([p.replace('_', ' ').title() for p in prompts])
        ax1.set_ylim(0, 110)
        ax1.legend(loc='upper right')

        # Add values on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            ax1.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_error_type_distribution(self, error_counts: Dict[str, int], filename: str = "error_type_distribution.png") -> str:
        """Plots error category distribution horizontal bar chart."""
        out_path = os.path.join(self.output_dir, filename)
        
        categories = list(error_counts.keys())
        counts = list(error_counts.values())

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        colors = ['#6366f1', '#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6']
        y_pos = range(len(categories))

        bars = ax.barh(y_pos, counts, color=colors[:len(categories)], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=9)
        ax.invert_yaxis()  # Top-down
        ax.set_xlabel('Number of Sentence Pairs', fontweight='bold')
        ax.set_title('Experiment 8: Grammar Error Category Distribution', fontsize=12, fontweight='bold', pad=15)

        for bar in bars:
            w = bar.get_width()
            ax.annotate(f'{w}', xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                        textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_metric_distributions(self, eval_df: pd.DataFrame, filename: str = "metric_distributions.png") -> str:
        """Plots distribution of Token F1 and Levenshtein Distance."""
        out_path = os.path.join(self.output_dir, filename)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

        # Token F1 Histogram
        ax1.hist(eval_df['token_f1'], bins=8, color='#6366f1', edgecolor='white', alpha=0.85)
        ax1.set_title('Token F1 Score Distribution', fontweight='bold')
        ax1.set_xlabel('Token F1 Score')
        ax1.set_ylabel('Sentence Count')

        # Levenshtein Distance Boxplot
        ax2.boxplot(eval_df['levenshtein_distance'], patch_artist=True,
                    boxprops=dict(facecolor='#10b981', color='black', alpha=0.8),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_title('Levenshtein Distance Distribution', fontweight='bold')
        ax2.set_ylabel('Character Edit Distance')
        ax2.set_xticklabels(['Evaluated Samples'])

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_over_vs_under_correction(self, eval_df: pd.DataFrame, filename: str = "over_vs_under_correction.png") -> str:
        """Plots comparison of Over-correction, Under-correction, and Exact Match counts."""
        out_path = os.path.join(self.output_dir, filename)
        
        total = len(eval_df)
        exact = int(eval_df['exact_match'].sum())
        over = int(eval_df['is_over_correction'].sum())
        under = int(eval_df['is_under_correction'].sum())
        other = total - (exact + over + under)
        if other < 0: other = 0

        labels = ['Exact Match', 'Over-Correction', 'Under-Correction', 'Valid Alternative Edit']
        values = [exact, over, under, other]
        colors = ['#10b981', '#f59e0b', '#ef4444', '#6366f1']

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Sentence Count', fontweight='bold')
        ax.set_title('Experiment 8: Correction Quality & Failure Mode Breakdown', fontsize=11, fontweight='bold', pad=15)
        
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
