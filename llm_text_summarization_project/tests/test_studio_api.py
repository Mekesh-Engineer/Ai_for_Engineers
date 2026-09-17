"""
Automated Test Suite for Local LLM Studio
Verifies dual-mode LLM abstraction, Ollama integration, Local model detection,
file summarization pipeline, project tree analysis, and FastAPI endpoints.
"""

import os
import sys
import unittest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.config.settings import settings
from backend.services.ollama_service import OllamaProvider
from backend.services.local_model_service import LocalModelProvider
from backend.services.file_service import file_service
from backend.services.summarizer import document_summarizer
from backend.services.project_analyzer import project_analyzer
from backend.services.llm_service import ChatMessage, get_llm_provider

class TestLocalLLMStudio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check_endpoint(self):
        """Verify /api/health responds with 200 and valid status payload."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["backend"])
        self.assertIn("active_mode", data)
        print("  [+] Test 01 Passed: /api/health is operational.")

    def test_02_models_overview_endpoint(self):
        """Verify /api/models returns Ollama and Local Model inspection."""
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("ollama", data)
        self.assertIn("local_model", data)
        print("  [+] Test 02 Passed: /api/models returned dual-mode overview.")

    def test_03_ollama_connectivity_and_test(self):
        """Verify Ollama provider reachability and diagnostic test."""
        ollama = OllamaProvider()
        is_online = asyncio.run(ollama.is_available())
        print(f"      Ollama service status: {'ONLINE' if is_online else 'OFFLINE'}")
        
        test_res = self.client.post("/api/models/test-ollama")
        self.assertEqual(test_res.status_code, 200)
        data = test_res.json()
        self.assertTrue(data["backend_running"])
        if is_online:
            self.assertTrue(data["ollama_reachable"])
            self.assertTrue(data["test_generation"])
            print("  [+] Test 03 Passed: Ollama connection and test generation succeeded.")
        else:
            print("  [!] Test 03 Notice: Ollama offline as expected in non-daemon env.")

    def test_04_local_model_discovery_and_test(self):
        """Verify local checkpoints in ./models/ are detected and validated."""
        local_prov = LocalModelProvider()
        models = local_prov.scan_models()
        self.assertGreater(len(models), 0, "Expected at least one local checkpoint in ./models/")
        
        test_res = self.client.post("/api/models/test-local")
        self.assertEqual(test_res.status_code, 200)
        data = test_res.json()
        self.assertTrue(data["model_detected"])
        self.assertTrue(data["model_valid"])
        self.assertTrue(data["test_generation"])
        print(f"  [+] Test 04 Passed: Local model validated ({models[0]['name']}) and generated output.")

    def test_05_model_mode_switching(self):
        """Verify switching modes via /api/models/switch."""
        # Switch to local_model
        res = self.client.post("/api/models/switch", json={"mode": "local_model"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(settings.active_mode, "local_model")

        # Switch to ollama
        res = self.client.post("/api/models/switch", json={"mode": "ollama", "model_name_or_path": "qwen2.5:7b"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(settings.active_mode, "ollama")
        print("  [+] Test 05 Passed: Hot model mode switching verified.")

    def test_06_file_text_extraction_and_chunking(self):
        """Verify file extraction and token chunking across document formats."""
        sample_code = """
import numpy as np

def compute_metrics(predictions, targets):
    '''Calculate standard regression error metrics.'''
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    return {'mse': mse, 'mae': mae}
"""
        doc = file_service.extract_text_from_bytes("metrics.py", sample_code.encode("utf-8"))
        self.assertEqual(doc.file_type, "PY")
        self.assertGreater(doc.word_count, 10)
        self.assertGreater(doc.estimated_tokens, 15)

        # Test chunking
        long_text = "\n\n".join([f"Paragraph {i}: " + "Machine learning algorithms optimize predictive functions. " * 30 for i in range(20)])
        chunks = file_service.chunk_text(long_text, max_chunk_tokens=500, overlap_tokens=50)
        self.assertGreater(len(chunks), 1)
        print(f"  [+] Test 06 Passed: File extraction and chunking verified ({len(chunks)} chunks).")

    def test_07_file_upload_api(self):
        """Verify /api/files/upload endpoint with multipart upload."""
        sample_content = b"# Architecture Overview\nThis project implements a dual-mode LLM system."
        res = self.client.post(
            "/api/files/upload",
            files={"file": ("readme_test.md", sample_content, "text/markdown")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document"]["filename"], "readme_test.md")
        print("  [+] Test 07 Passed: /api/files/upload successfully processed document.")

    def test_08_project_tree_and_scanner(self):
        """Verify /api/project/tree and /api/project/files."""
        tree_res = self.client.get("/api/project/tree")
        self.assertEqual(tree_res.status_code, 200)
        tree = tree_res.json()
        self.assertTrue(tree["is_dir"])
        self.assertIn("children", tree)

        files_res = self.client.get("/api/project/files")
        self.assertEqual(files_res.status_code, 200)
        files = files_res.json()
        self.assertGreater(len(files), 0)
        print(f"  [+] Test 08 Passed: Project tree and context file scanner verified ({len(files)} files).")

    def test_09_settings_and_profiles(self):
        """Verify /api/settings and custom profile creation."""
        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("profiles", data)
        self.assertIn("general", data["profiles"])
        self.assertIn("coding", data["profiles"])

        # Test custom profile creation
        prof_res = self.client.post("/api/settings/profile", json={
            "profile_id": "test_specialist",
            "name": "Test Specialist",
            "description": "Custom automated test persona",
            "prompt": "You are a test engineer."
        })
        self.assertEqual(prof_res.status_code, 200)
        print("  [+] Test 09 Passed: Settings and profile management verified.")

    def test_10_chat_and_debugger_endpoints(self):
        """Verify chat endpoint with project context and debugger assistant."""
        # Test chat non-streaming
        chat_res = self.client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Explain what this project does in 1 sentence."}],
            "mode": "local_model"
        })
        self.assertEqual(chat_res.status_code, 200)
        chat_data = chat_res.json()
        self.assertIn("conversation_id", chat_data)
        self.assertIn("response", chat_data)

        # Test debugger endpoint
        debug_res = self.client.post("/api/project/debug", json={
            "error_message": "IndexError: list index out of range",
            "stack_trace": "File main.py, line 42, in process_tokens",
            "mode": "local_model"
        })
        self.assertEqual(debug_res.status_code, 200)
        debug_data = debug_res.json()
        self.assertIn("diagnosis", debug_data)
        print("  [+] Test 10 Passed: Chat and Debugger endpoints operational.")

if __name__ == "__main__":
    unittest.main()
