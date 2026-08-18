"""AQuA loader (https://github.com/google-deepmind/AQuA).

Expects the official AQuA dev split at `raps_data/aqua/dev.json` — one JSON
object per line with fields {"question", "options" (["A) ...", ...]),
"rationale", "correct"}. The dev split is the 254-instance subset used by the
agent-evaluation protocol of GPTSwarm/G-Designer (Supp. Table S.3). Answers are
option letters, so extraction mirrors the MMLU postprocessing.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from RAPS.utils.const import RAPS_ROOT

DATA_PATH = RAPS_ROOT / "raps_data/aqua/dev.json"


def aqua_data_process(records: List[Dict]) -> List[Dict]:
    items = []
    for r in records:
        options = "\n".join(
            f"Option {o[0]}: {o[2:].strip()}" if re.match(r"^[A-E]\)", o) else f"Option {o}"
            for o in r["options"]
        )
        items.append({
            "task": f"{r['question'].strip()}\n{options}",
            "answer": r["correct"].strip(),
        })
    return items


def load_aqua(path: Optional[str] = None) -> List[Dict]:
    data_path = Path(path) if path else DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"AQuA data not found at {data_path}. Download dev.json from "
            "https://github.com/google-deepmind/AQuA into raps_data/aqua/."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        return aqua_data_process([json.loads(line) for line in f if line.strip()])


def aqua_get_predict(text) -> str:
    """MMLU-style extraction: the letter after 'answer is', else the first
    standalone option letter in the string. Returns '' when nothing is found."""
    if isinstance(text, list):
        text = text[0] if text else ""
    if not isinstance(text, str) or not text:
        return ""
    pos = text.find("answer is")
    if pos != -1:
        tail = text[pos + len("answer is"):].strip(":").strip().strip("Option").strip()
        if tail:
            return tail[0].upper()
    m = re.search(r"\b([A-E])\b", text)
    return m.group(1) if m else ""
