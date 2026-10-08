import os
from typing import Any, Dict, List

from .core import Agent, Task, Result


class SearchAgent:
    name = "search"
    description = "Search files by name and simple content"
    capabilities = ["search"]

    def run(self, task: Task) -> Result:
        query = str(task.params.get("query", "")).lower()
        matched: List[str] = []
        for root, _, files in os.walk(os.getcwd()):
            for f in files:
                full = os.path.join(root, f)
                if query and f.lower().find(query) >= 0:
                    matched.append(full)
                    continue
                if query:
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read().lower()
                            if content.find(query) >= 0:
                                matched.append(full)
                    except Exception:
                        pass
        return Result(True, f"Found {len(matched)} files", matched)


class CodeSurveyAgent:
    name = "code-survey"
    description = "Summarize repository by file extensions"
    capabilities = ["survey"]

    def run(self, task: Task) -> Result:
        counts: Dict[str, int] = {}
        for root, _, files in os.walk(os.getcwd()):
            for f in files:
                ext = os.path.splitext(f)[1] or "none"
                counts[ext] = counts.get(ext, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        return Result(True, summary, counts)


class PlanAgent:
    name = "plan"
    description = "Generate a simple step-by-step plan"
    capabilities = ["plan"]

    def run(self, task: Task) -> Result:
        steps = [
            "Understand requirements",
            "Choose stack",
            "Design modules",
            "Implement features",
            "Test outputs",
        ]
        return Result(True, task.description, steps)