"""Shared helpers for the analysis scripts: task-log loading and token estimates."""
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Allow running as `python analysis/<script>.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # tiktoken optional
    _ENC = None


def load_task_logs(pattern: str) -> List[Dict[str, Any]]:
    """Load per-task trace JSONs (logs/log_<domain>_*.json)."""
    logs = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                log = json.load(f)
            log["_path"] = os.path.basename(path)
            logs.append(log)
        except (json.JSONDecodeError, OSError):
            continue
    return logs


def count_tokens(text: Any) -> int:
    """Approximate token count of a message payload (cl100k, else chars/4).

    Task logs store message text rather than API usage fields, so the
    per-mechanism decomposition is an estimate; the exact totals live in the
    runners' global counters (utils/globals.py)."""
    s = str(text)
    if _ENC is not None:
        return len(_ENC.encode(s))
    return max(1, len(s) // 4)


def iter_llm_traces(task_log: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield every llm_trace entry of every step, tagged with its call type
    (publish / refine_system_prompt / broker_route / watchdog_evaluate / ...)."""
    for step in task_log.get("steps", []):
        for agent_id, traces in step.get("llm_calls", {}).items():
            for tr in traces:
                yield {"step": step.get("step"), "agent": agent_id,
                       "type": tr.get("type", "unknown"),
                       "messages": tr.get("messages", []),
                       "output": tr.get("output", "")}
