# -*- coding: utf-8 -*-
"""
verify_test_case.py
-------------------
Comprehensive test suite and automated verification for Experiment 7:
LLM Text Summarization System. Tests all modules, edge cases, prompt templates,
summarizers, evaluation formulas, and result exports.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to python path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader import (
    load_config,
    get_or_create_raw_articles,
    split_dataset,
    save_processed_data,
    load_test_documents,
    load_reference_summaries
)
from data_preprocessing import (
    clean_text,
    tokenize_sentences,
    compute_text_statistics,
    filter_documents_by_length,
    preprocess_corpus,
    verify_data_integrity
)
from prompt_templates import (
    PromptManager,
    create_summarization_prompt,
    BUILTIN_PROMPTS
)
from summarizer import (
    AbstractiveSummarizer,
    ExtractiveSummarizer,
    summarize_with_prompt_engineering
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


class TestTextSummarizationPipeline(unittest.TestCase):
    """Unit and Integration Tests for Experiment 7."""

    @classmethod
    def setUpClass(cls):
        cls.test_doc = (
            "Artificial Intelligence (AI) and Machine Learning (ML) have revolutionized modern engineering systems "
            "across multiple disciplines. In civil and structural engineering, automated computer vision models are "
            "now deployed on unmanned aerial vehicles (UAVs) to perform real-time structural health monitoring, "
            "detecting micro-cracks in concrete bridges before catastrophic failures occur. In electrical and power "
            "systems engineering, recurrent neural networks forecast smart grid load demands, balancing renewable "
            "energy integration with traditional thermal generation. Meanwhile, mechanical engineers utilize deep "
            "reinforcement learning algorithms to optimize thermal management systems in electric vehicle battery packs, "
            "extending range and battery operational lifespan."
        )
        cls.test_ref = (
            "AI and ML technologies are transforming civil, electrical, and mechanical engineering through automated "
            "structural monitoring, smart grid power forecasting, and EV battery thermal optimization."
        )

    # ── Test Group 1: Configuration & Environment ──────────────────────────
    def test_01_config_loading(self):
        cfg_path = os.path.join(_PROJECT_ROOT, "config", "model_config.json")
        cfg = load_config(cfg_path)
        self.assertIn("model", cfg)
        self.assertIn("generation", cfg)
        self.assertIn("dataset", cfg)
        self.assertIn("evaluation", cfg)
        self.assertGreater(cfg["generation"]["max_length"], cfg["generation"]["min_length"])

    # ── Test Group 2: Data Loader & Preprocessing ──────────────────────────
    def test_02_corpus_generation_and_loading(self):
        raw_csv = os.path.join(_PROJECT_ROOT, "data", "raw", "articles.csv")
        df = get_or_create_raw_articles(raw_csv)
        self.assertFalse(df.empty)
        self.assertIn("document_text", df.columns)
        self.assertIn("reference_summary", df.columns)
        self.assertGreaterEqual(len(df), 5)

    def test_03_text_cleaning_and_normalization(self):
        dirty_text = "   <div>Artificial Intelligence &nbsp; is <b>great</b>!</div> \n\n It's   revolutionary.  "
        cleaned = clean_text(dirty_text)
        self.assertNotIn("<div>", cleaned)
        self.assertNotIn("&nbsp;", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertEqual(cleaned, "Artificial Intelligence is great! It's revolutionary.")

    def test_04_sentence_tokenization(self):
        sentences = tokenize_sentences(self.test_doc)
        self.assertIsInstance(sentences, list)
        self.assertGreaterEqual(len(sentences), 3)

    def test_05_data_integrity_and_splitting(self):
        raw_csv = os.path.join(_PROJECT_ROOT, "data", "raw", "articles.csv")
        df = get_or_create_raw_articles(raw_csv)
        processed = preprocess_corpus(df)
        integrity = verify_data_integrity(processed)
        self.assertEqual(integrity["status"], "PASSED")
        self.assertEqual(integrity["missing_documents"], 0)

        train, val, test = split_dataset(processed, 0.8, 0.1, 0.1)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    # ── Test Group 3: Prompt Engineering Module ───────────────────────────
    def test_06_prompt_manager_and_templates(self):
        prompt_dir = os.path.join(_PROJECT_ROOT, "models", "prompts")
        pm = PromptManager(prompt_dir=prompt_dir)
        keys = pm.get_prompt_keys()
        self.assertIn("v1", keys)
        self.assertIn("v2", keys)
        self.assertIn("best", keys)

        formatted_v1 = pm.format_prompt("v1", self.test_doc)
        self.assertIn(self.test_doc, formatted_v1)
        self.assertIn("2 to 3 sentences", formatted_v1)

        formatted_best = create_summarization_prompt(self.test_doc, "best", prompt_dir)
        self.assertIn("CORE TAKEAWAY", formatted_best)

    # ── Test Group 4: Summarization Models ──────────────────────────────────
    def test_07_extractive_summarizer(self):
        extractive = ExtractiveSummarizer(num_sentences=2)
        summary = extractive.summarize(self.test_doc)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary.split()), 10)
        self.assertLess(len(summary.split()), len(self.test_doc.split()))
        # Verify extracted words are from source
        words = summary.split()
        self.assertTrue(any(w in self.test_doc for w in words[:5]))

    def test_08_abstractive_summarizer_inference(self):
        summarizer = AbstractiveSummarizer(model_name="sshleifer/distilbart-cnn-12-6", device="cpu")
        summary = summarizer.summarize(self.test_doc, min_length=20, max_length=60, num_beams=2)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary.split()), 5)
        self.assertLess(len(summary.split()), len(self.test_doc.split()))

    def test_09_prompt_engineered_summarization(self):
        res = summarize_with_prompt_engineering(
            self.test_doc,
            prompt_template_key="best",
            generation_kwargs={"min_length": 20, "max_length": 70, "num_beams": 2}
        )
        self.assertIn("summary_text", res)
        self.assertIn("latency_seconds", res)
        self.assertGreater(res["summary_word_count"], 5)

    # ── Test Group 5: Evaluation Metrics ───────────────────────────────────
    def test_10_rouge_calculation(self):
        scores = calculate_rouge_scores(self.test_ref, self.test_ref)
        # Identical text should have F1 = 1.0
        self.assertAlmostEqual(scores["rouge1"]["f1"], 1.0, delta=0.01)
        self.assertAlmostEqual(scores["rouge2"]["f1"], 1.0, delta=0.01)
        self.assertAlmostEqual(scores["rougeL"]["f1"], 1.0, delta=0.01)

    def test_11_bertscore_and_compression(self):
        bert = calculate_bertscore([self.test_ref], [self.test_ref])
        self.assertGreaterEqual(bert["f1"], 0.85)

        comp = calculate_compression_ratio(self.test_doc, self.test_ref)
        self.assertGreater(comp["word_compression_percentage"], 50.0)

        length_chk = assess_length_compliance(self.test_ref, target_length_words=25, tolerance_pct=40.0)
        self.assertIn("is_compliant", length_chk)

    def test_12_evaluation_report_generation(self):
        sample_df = pd.DataFrame([{
            "id": "DOC_TEST",
            "rouge1_f1": 0.85,
            "rouge2_f1": 0.72,
            "rougeL_f1": 0.81,
            "bertscore_f1": 0.92,
            "word_compression_pct": 76.5,
            "length_compliant": True
        }])
        report = generate_evaluation_report(sample_df)
        self.assertIn("EXPERIMENT 7: LLM TEXT SUMMARIZATION", report)
        self.assertIn("DOC_TEST", report)

    # ── Test Group 6: Visualizations ───────────────────────────────────────
    def test_13_visualization_exports(self):
        test_viz_dir = os.path.join(_PROJECT_ROOT, "results", "test_viz")
        os.makedirs(test_viz_dir, exist_ok=True)

        prompt_df = pd.DataFrame([
            {"prompt_version": "Prompt v1", "rouge1_f1": 0.78, "rouge2_f1": 0.65, "rougeL_f1": 0.74, "bertscore_f1": 0.88, "latency_seconds": 1.2},
            {"prompt_version": "Prompt v2", "rouge1_f1": 0.81, "rouge2_f1": 0.68, "rougeL_f1": 0.77, "bertscore_f1": 0.90, "latency_seconds": 1.4},
            {"prompt_version": "Prompt best", "rouge1_f1": 0.86, "rouge2_f1": 0.74, "rougeL_f1": 0.82, "bertscore_f1": 0.93, "latency_seconds": 1.5}
        ])
        p1 = os.path.join(test_viz_dir, "test_prompt.png")
        plot_prompt_comparison(prompt_df, p1)
        self.assertTrue(os.path.exists(p1))

        eval_df = pd.DataFrame([
            {
                "id": "DOC_001",
                "rouge1_precision": 0.80, "rouge1_recall": 0.85, "rouge1_f1": 0.82,
                "rouge2_precision": 0.65, "rouge2_recall": 0.70, "rouge2_f1": 0.67,
                "rougeL_precision": 0.75, "rougeL_recall": 0.80, "rougeL_f1": 0.77,
                "bertscore_f1": 0.91, "summary_words": 35, "doc_word_count": 140,
                "word_compression_pct": 75.0, "length_compliant": True
            }
        ])
        p2 = os.path.join(test_viz_dir, "test_dist.png")
        plot_rouge_score_distributions(eval_df, p2)
        self.assertTrue(os.path.exists(p2))

        p3 = os.path.join(test_viz_dir, "test_comp.png")
        plot_length_and_compression(eval_df, p3)
        self.assertTrue(os.path.exists(p3))

        abs_ext_df = pd.DataFrame([
            {"approach": "Abstractive (BART)", "rouge1_f1": 0.82, "rouge2_f1": 0.68, "rougeL_f1": 0.79, "bertscore_f1": 0.92},
            {"approach": "Extractive (TextRank)", "rouge1_f1": 0.71, "rouge2_f1": 0.54, "rougeL_f1": 0.66, "bertscore_f1": 0.85}
        ])
        p4 = os.path.join(test_viz_dir, "test_abs_ext.png")
        plot_abstractive_vs_extractive(abs_ext_df, p4)
        self.assertTrue(os.path.exists(p4))


def run_tests():
    print("=" * 75)
    print("RUNNING COMPREHENSIVE TEST SUITE FOR EXPERIMENT 7")
    print("=" * 75)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextSummarizationPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[+] ALL 13 TEST CASES PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n[!] TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
