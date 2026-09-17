#!/usr/bin/env python3
"""
Automated Verification & Unit Test Suite for Experiment 8
Validates all data loaders, preprocessing, prompt templates, evaluation metrics, and correction modules.
"""

import os
import unittest
import pandas as pd
from src.data_loader import DataLoaderModule
from src.data_preprocessing import PreprocessingModule
from src.prompt_templates import PromptTemplates, PromptManager
from src.evaluation import EvaluationModule, compute_levenshtein_distance
from src.error_analyzer import ErrorAnalyzerModule
from src.corrector import GrammarCorrector


class TestExperiment8GrammarCorrection(unittest.TestCase):
    
    def setUp(self):
        self.data_loader = DataLoaderModule()
        self.preprocessor = PreprocessingModule()
        self.prompt_manager = PromptManager()
        self.evaluator = EvaluationModule()
        self.analyzer = ErrorAnalyzerModule()
        self.corrector = GrammarCorrector()

    def test_01_environment_and_imports(self):
        """Test 01: Verify core modules import cleanly."""
        self.assertIsNotNone(self.data_loader)
        self.assertIsNotNone(self.preprocessor)
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.corrector)
        print("  [+] Test 01 Passed: Environment and core modules initialized.")

    def test_02_data_loader_and_integrity(self):
        """Test 02: Verify dataset loading and integrity check."""
        df = self.data_loader.load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 10)
        self.assertIn("error_sentence", df.columns)
        self.assertIn("corrected_sentence", df.columns)

        stats = self.data_loader.verify_data_integrity(df)
        self.assertTrue(stats["integrity_passed"])
        self.assertGreater(stats["avg_error_words"], 0)
        print("  [+] Test 02 Passed: Data loader and data integrity validated.")

    def test_03_data_preprocessing(self):
        """Test 03: Verify whitespace cleaning and punctuation normalization."""
        raw = "  Although  it was raining  heavy , but we continued .  "
        cleaned = self.preprocessor.normalize_punctuation(raw)
        self.assertEqual(cleaned, "Although it was raining heavy, but we continued.")

        quote_raw = "\u201CHe go to lab\u201D"
        cleaned_quote = self.preprocessor.clean_text(quote_raw)
        self.assertEqual(cleaned_quote, '"He go to lab"')
        print("  [+] Test 03 Passed: Data preprocessing and text normalization verified.")

    def test_04_dataset_split(self):
        """Test 04: Verify dataset train/val/test splitting."""
        df = self.data_loader.load_raw_data()
        train, val, test = self.data_loader.split_dataset(df, 0.8, 0.1, 0.1)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(val) + len(test), len(df))
        print("  [+] Test 04 Passed: Train/validation/test split validated.")

    def test_05_error_categorization(self):
        """Test 05: Verify error type distribution calculation."""
        df = self.data_loader.load_raw_data()
        counts = self.data_loader.categorize_errors(df)
        self.assertIsInstance(counts, dict)
        self.assertGreater(len(counts), 0)
        print(f"  [+] Test 05 Passed: Error categorization verified ({len(counts)} categories).")

    def test_06_prompt_templates(self):
        """Test 06: Verify prompt formatting across multi-level templates."""
        sample = "He go to school."
        for version in ["minimal", "standard", "rewrite", "detection", "academic"]:
            formatted = PromptTemplates.format_prompt(version, sample)
            self.assertIn(sample, formatted)
            self.assertIsInstance(formatted, str)
        print("  [+] Test 06 Passed: Multi-level prompt templates verified.")

    def test_07_prompt_manager(self):
        """Test 07: Verify PromptManager custom registration and version selection."""
        pm = PromptManager(default_version="minimal")
        self.assertEqual(pm.active_version, "minimal")
        
        pm.register_template("custom_esl", "Please fix ESL grammar: {text}\nFixed:")
        formatted = pm.get_prompt("She like apple.", version="custom_esl")
        self.assertEqual(formatted, "Please fix ESL grammar: She like apple.\nFixed:")
        print("  [+] Test 07 Passed: PromptManager and template customization verified.")

    def test_08_levenshtein_distance(self):
        """Test 08: Verify Levenshtein edit distance computation."""
        s1 = "kitten"
        s2 = "sitting"
        # kitten -> sitten -> sittin -> sitting (3 edits)
        dist = compute_levenshtein_distance(s1, s2)
        self.assertEqual(dist, 3)

        # Identical strings
        self.assertEqual(compute_levenshtein_distance("hello", "hello"), 0)
        # Empty string distance
        self.assertEqual(compute_levenshtein_distance("", "test"), 4)
        print("  [+] Test 08 Passed: Levenshtein edit distance algorithm validated.")

    def test_09_token_f1_score(self):
        """Test 09: Verify Token-level Precision, Recall, and F1 calculations."""
        # Identical tokens
        res_exact = self.evaluator.calculate_token_f1("The quick brown fox", "The quick brown fox")
        self.assertEqual(res_exact["f1"], 1.0)

        # Partial overlap: "The data is valid" vs "The data are valid" -> 3 out of 4 common
        res_part = self.evaluator.calculate_token_f1("The data is valid", "The data are valid")
        self.assertEqual(res_part["precision"], 0.75)
        self.assertEqual(res_part["recall"], 0.75)
        self.assertEqual(res_part["f1"], 0.75)
        print("  [+] Test 09 Passed: Token-level F1 precision and recall metrics verified.")

    def test_10_exact_match_calculation(self):
        """Test 10: Verify boolean Exact Match accuracy."""
        self.assertTrue(self.evaluator.calculate_exact_match("He went home.", "he went home."))
        self.assertTrue(self.evaluator.calculate_exact_match("  Data is clean. ", "Data is clean."))
        self.assertFalse(self.evaluator.calculate_exact_match("Data is clean", "Data are clean"))
        print("  [+] Test 10 Passed: Exact match scoring logic validated.")

    def test_11_over_and_under_correction_detection(self):
        """Test 11: Verify detection of over-correction and under-correction."""
        input_sent = "He go to laboratory."
        ref_sent = "He went to the laboratory."
        
        # Under-correction: unchanged input
        over, under = self.evaluator.detect_over_and_under_correction(input_sent, input_sent, ref_sent)
        self.assertTrue(under)
        self.assertFalse(over)

        # Correct edit
        over_c, under_c = self.evaluator.detect_over_and_under_correction(input_sent, ref_sent, ref_sent)
        self.assertFalse(over_c)
        self.assertFalse(under_c)
        print("  [+] Test 11 Passed: Over-correction and under-correction heuristics validated.")

    def test_12_grammar_corrector_inference(self):
        """Test 12: Verify grammar corrector execution (single and batch)."""
        sample = "He go to the laboratory yesterday for doing the experiment."
        expected = "He went to the laboratory yesterday to do the experiment."
        pred = self.corrector.correct_sentence(sample, mode="standard")
        self.assertEqual(pred.strip(), expected.strip())

        # Batch
        batch_res = self.corrector.batch_correct([sample], mode="standard")
        self.assertEqual(len(batch_res), 1)
        self.assertEqual(batch_res[0].strip(), expected.strip())
        print("  [+] Test 12 Passed: GrammarCorrector pipeline execution verified.")

    def test_13_error_analyzer_and_reports(self):
        """Test 13: Verify error type breakdown and diagnostic report generation."""
        inputs = ["He go to school.", "The datas was corrupt."]
        preds = ["He went to school.", "The data was corrupt."]
        refs = ["He went to school.", "The data was corrupted."]
        types = ["Verb Tense", "Pluralization"]

        _, eval_df = self.evaluator.evaluate_batch(inputs, preds, refs, types)
        grouped = self.analyzer.analyze_by_error_type(eval_df)
        self.assertFalse(grouped.empty)
        
        report = self.analyzer.generate_diagnostic_summary(eval_df)
        self.assertIn("GRAMMAR ERROR CORRECTION", report)
        print("  [+] Test 13 Passed: ErrorAnalyzerModule and diagnostic reports validated.")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("   RUNNING EXPERIMENT 8 AUTOMATED VALIDATION TEST SUITE   ")
    print("=" * 65 + "\n")
    unittest.main(verbosity=2)
