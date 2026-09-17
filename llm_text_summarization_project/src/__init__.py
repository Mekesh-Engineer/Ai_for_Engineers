# -*- coding: utf-8 -*-
"""
LLM Text Summarization System Package
Experiment 7: Text Summarization using Large Language Models
"""

from .data_loader import (
    load_config,
    get_or_create_raw_articles,
    split_dataset,
    save_processed_data,
    load_test_documents,
    load_reference_summaries,
    save_generated_summaries
)

from .data_preprocessing import (
    clean_text,
    tokenize_sentences,
    compute_text_statistics,
    filter_documents_by_length,
    preprocess_corpus,
    verify_data_integrity
)

from .prompt_templates import (
    PromptManager,
    create_summarization_prompt,
    BUILTIN_PROMPTS
)

from .summarizer import (
    AbstractiveSummarizer,
    ExtractiveSummarizer,
    summarize_with_prompt_engineering
)

from .model_trainer import (
    train_or_finetune_summarizer,
    save_model_artifacts
)

from .evaluation import (
    calculate_rouge_scores,
    calculate_bertscore,
    assess_length_compliance,
    calculate_compression_ratio,
    evaluate_summary_quality,
    generate_evaluation_report
)

from .visualization import (
    plot_prompt_comparison,
    plot_rouge_score_distributions,
    plot_length_and_compression,
    plot_abstractive_vs_extractive
)

__all__ = [
    "load_config",
    "get_or_create_raw_articles",
    "split_dataset",
    "save_processed_data",
    "load_test_documents",
    "load_reference_summaries",
    "save_generated_summaries",
    "clean_text",
    "tokenize_sentences",
    "compute_text_statistics",
    "filter_documents_by_length",
    "preprocess_corpus",
    "verify_data_integrity",
    "PromptManager",
    "create_summarization_prompt",
    "BUILTIN_PROMPTS",
    "AbstractiveSummarizer",
    "ExtractiveSummarizer",
    "summarize_with_prompt_engineering",
    "train_or_finetune_summarizer",
    "save_model_artifacts",
    "calculate_rouge_scores",
    "calculate_bertscore",
    "assess_length_compliance",
    "calculate_compression_ratio",
    "evaluate_summary_quality",
    "generate_evaluation_report",
    "plot_prompt_comparison",
    "plot_rouge_score_distributions",
    "plot_length_and_compression",
    "plot_abstractive_vs_extractive"
]
