import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config.settings import settings
from backend.services.llm_service import LLMProvider, get_llm_provider
from backend.utils.logger import studio_logger

IGNORED_DIRS = {
    ".git", ".vscode", ".idea", "__pycache__", "node_modules",
    ".venv", "venv", "env", "dist", "build", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "site-packages"
}

IGNORED_EXTENSIONS = {
    ".pt", ".safetensors", ".bin", ".pyc", ".pyd", ".exe",
    ".dll", ".so", ".dylib", ".png", ".jpg", ".jpeg", ".gif",
    ".zip", ".tar", ".gz", ".7z", ".mp4", ".mp3", ".wav"
}

class ProjectAnalyzer:
    """Scans and analyzes local workspace repositories, dependencies, architecture, docstrings, and grammar health."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or settings.project_root).resolve()

    def get_directory_tree(self, max_depth: int = 4) -> Dict[str, Any]:
        """Build structured file tree dictionary for the frontend."""
        def _build_node(path: Path, current_depth: int) -> Dict[str, Any]:
            rel_path = str(path.relative_to(self.root_dir))
            is_dir = path.is_dir()
            
            node = {
                "name": path.name if path != self.root_dir else self.root_dir.name,
                "path": str(path),
                "relative_path": "." if path == self.root_dir else rel_path.replace("\\", "/"),
                "is_dir": is_dir,
                "size_bytes": 0 if is_dir else path.stat().st_size
            }

            if is_dir:
                children = []
                if current_depth < max_depth:
                    try:
                        entries = sorted(list(path.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
                        for entry in entries:
                            if entry.name in IGNORED_DIRS:
                                continue
                            if not entry.is_dir() and entry.suffix.lower() in IGNORED_EXTENSIONS:
                                continue
                            children.append(_build_node(entry, current_depth + 1))
                    except PermissionError:
                        pass
                node["children"] = children
            return node

        return _build_node(self.root_dir, 0)

    def scan_project_files(self) -> List[Dict[str, Any]]:
        """List all text/code files available for context selection."""
        file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for f in files:
                file_path = Path(root) / f
                if file_path.suffix.lower() in IGNORED_EXTENSIONS:
                    continue
                try:
                    size = file_path.stat().st_size
                    rel_path = str(file_path.relative_to(self.root_dir)).replace("\\", "/")
                    file_list.append({
                        "filename": f,
                        "path": str(file_path),
                        "relative_path": rel_path,
                        "extension": file_path.suffix.lower(),
                        "size_bytes": size,
                        "is_important": f in ["README.md", "main.py", "requirements.txt", "package.json", "pyproject.toml", "config.json", "model_config.json"]
                    })
                except Exception:
                    continue
        
        return sorted(file_list, key=lambda x: (not x["is_important"], x["relative_path"]))

    def read_file_content(self, relative_path: str, max_chars: int = 15000) -> str:
        """Safely read content of a specific file in the repository."""
        target_path = (self.root_dir / relative_path).resolve()
        try:
            target_path.relative_to(self.root_dir)
        except ValueError:
            raise PermissionError("Access denied: Path traversal detected outside project root.")

        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")

        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
                if len(content) == max_chars:
                    content += "\n... [Content truncated for context safety] ..."
                return content
        except Exception as e:
            return f"[Error reading file {relative_path}: {str(e)}]"

    def extract_dependencies(self) -> Dict[str, Any]:
        """Extract project dependencies from requirements.txt, pyproject.toml, package.json."""
        deps = {
            "python": [],
            "node": [],
            "config_files": []
        }

        req_path = self.root_dir / "requirements.txt"
        if req_path.exists():
            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        clean = line.strip()
                        if clean and not clean.startswith("#"):
                            deps["python"].append(clean)
            except Exception:
                pass

        pkg_path = self.root_dir / "package.json"
        if pkg_path.exists():
            try:
                with open(pkg_path, "r", encoding="utf-8") as f:
                    pkg_data = json.load(f)
                    deps["node"] = list(pkg_data.get("dependencies", {}).keys()) + list(pkg_data.get("devDependencies", {}).keys())
            except Exception:
                pass

        for cfg_name in ["model_config.json", "api_config.json", "prompts.json", "config.json"]:
            if (self.root_dir / "config" / cfg_name).exists() or (self.root_dir / cfg_name).exists():
                deps["config_files"].append(cfg_name)

        return deps

    def perform_heuristic_checks(self) -> List[Dict[str, Any]]:
        """Perform static rule-based sanity checks on project health and grammar pipeline assets."""
        checks = []

        # Check README
        readme_exists = any((self.root_dir / name).exists() for name in ["README.md", "readme.md", "README.txt"])
        checks.append({
            "check": "Project Documentation (README.md)",
            "passed": readme_exists,
            "status": "Healthy" if readme_exists else "Warning",
            "message": "README documentation found." if readme_exists else "Missing README.md file."
        })

        # Check Dependencies
        req_exists = (self.root_dir / "requirements.txt").exists() or (self.root_dir / "pyproject.toml").exists()
        checks.append({
            "check": "Dependency Manifest",
            "passed": req_exists,
            "status": "Healthy" if req_exists else "Critical",
            "message": "Python dependencies declared." if req_exists else "No requirements.txt or pyproject.toml found."
        })

        # Check Model Checkpoints
        models_dir = self.root_dir / "models"
        has_models = models_dir.exists() and (any(models_dir.rglob("*.safetensors")) or any(models_dir.rglob("*.pt")) or any(models_dir.rglob("*.bin")))
        checks.append({
            "check": "Local Model Storage",
            "passed": has_models,
            "status": "Healthy" if has_models else "Info",
            "message": "Local model weights detected in ./models/." if has_models else "No local model weights found in ./models/."
        })

        # Check Dataset
        data_dir = self.root_dir / "data" / "raw"
        has_dataset = data_dir.exists() and any(data_dir.glob("*.csv"))
        checks.append({
            "check": "Grammar Error Datasets",
            "passed": has_dataset,
            "status": "Healthy" if has_dataset else "Warning",
            "message": "Grammar error datasets found in data/raw/." if has_dataset else "Missing raw error datasets."
        })

        return checks

    async def analyze_project(
        self,
        selected_files: Optional[List[str]] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive LLM-driven architectural, code documentation, and grammar audit analysis."""
        provider = get_llm_provider(mode)
        tree = self.get_directory_tree(max_depth=3)
        deps = self.extract_dependencies()
        heuristics = self.perform_heuristic_checks()

        # Gather key files context
        files_to_read = selected_files or [
            "README.md", "main.py", "requirements.txt", "config/model_config.json"
        ]

        file_contexts = []
        for rel in files_to_read:
            path = self.root_dir / rel
            if path.exists() and path.is_file():
                content = self.read_file_content(rel, max_chars=4000)
                file_contexts.append(f"=== File: {rel} ===\n{content}")

        aggregated_context = "\n\n".join(file_contexts)
        
        prompt = (
            "Analyze this software repository thoroughly based on the project manifest and key source files.\n\n"
            f"Dependencies:\n{json.dumps(deps, indent=2)}\n\n"
            f"Source Files & Context:\n{aggregated_context}\n\n"
            "Provide a comprehensive technical report formatted in Markdown with the following sections:\n"
            "1. **Project Overview & Architecture**: Core objective (Grammar Correction & Text Rewriting), design patterns, and module separation.\n"
            "2. **Key Components & Flow**: Important files, APIs, dataset pipelines, and correction models.\n"
            "3. **Documentation & Docstring Grammar Review**: Evaluation of comments, explanations, and clarity.\n"
            "4. **Potential Problems & Code Smells**: Bottlenecks, missing validations, or edge cases.\n"
            "5. **Strategic Recommendations**: Concrete next steps for grammar correction enhancement and system scaling."
        )

        res = await provider.generate(
            prompt=prompt,
            system_prompt=settings.get_profile_prompt("technical_editor"),
            temperature=0.2
        )

        return {
            "mode_used": provider.mode_name,
            "directory_tree": tree,
            "dependencies": deps,
            "heuristic_checks": heuristics,
            "analyzed_files": files_to_read,
            "ai_report": res.text,
            "duration_seconds": res.duration_seconds
        }

    async def debug_issue(
        self,
        error_message: str,
        stack_trace: str,
        relevant_file: Optional[str] = None,
        expected_behavior: Optional[str] = None,
        actual_behavior: Optional[str] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Diagnose a software or model error using the AI debugging assistant."""
        provider = get_llm_provider(mode)
        
        file_context = ""
        if relevant_file:
            try:
                file_context = self.read_file_content(relevant_file, max_chars=5000)
            except Exception as e:
                file_context = f"[Could not read file {relevant_file}: {e}]"

        prompt = (
            f"Diagnose the following issue:\n\n"
            f"Error / Symptom:\n{error_message}\n\n"
            f"Stack Trace:\n```\n{stack_trace}\n```\n\n"
            f"Relevant File ({relevant_file or 'N/A'}):\n```\n{file_context}\n```\n\n"
            f"Expected Behavior:\n{expected_behavior or 'Normal operation'}\n\n"
            f"Actual Behavior:\n{actual_behavior or error_message}\n\n"
            "Provide a complete diagnosis containing:\n"
            "1. **Root Cause Analysis**: Why did this fail?\n"
            "2. **Exact Fix**: Step-by-step resolution and code diff/snippet.\n"
            "3. **Defensive Recommendations**: How to prevent similar issues in the future."
        )

        res = await provider.generate(
            prompt=prompt,
            system_prompt="You are an expert AI software and ML debugging assistant. Diagnose errors with clear root cause analyses and precise code fixes.",
            temperature=0.2
        )

        return {
            "mode_used": provider.mode_name,
            "diagnosis": res.text,
            "duration_seconds": res.duration_seconds
        }

project_analyzer = ProjectAnalyzer()
