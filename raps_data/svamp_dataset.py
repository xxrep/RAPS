"""SVAMP loader (https://github.com/arkilpatel/SVAMP).

Expects the official `SVAMP.json` at `raps_data/svamp/SVAMP.json` — a single
JSON list of objects with fields {"ID", "Body", "Question", "Equation",
"Answer", "Type"}. The task text is `Body + " " + Question`; the answer is the
numeric `Answer` field. Extraction reuses the GSM8K answer parser, since both
benchmarks score a single number (Supp. Table S.3: 1,000 instances).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from RAPS.utils.const import RAPS_ROOT
from raps_data.gsm8k_dataset import gsm_get_predict  # re-exported for runners

DATA_PATH = RAPS_ROOT / "raps_data/svamp/SVAMP.json"


def svamp_data_process(records: List[Dict]) -> List[Dict]:
    return [
        {
            "task": f"{r['Body'].strip()} {r['Question'].strip()}",
            "answer": str(r["Answer"]),
            "id": r.get("ID", ""),
        }
        for r in records
    ]


def load_svamp(path: Optional[str] = None) -> List[Dict]:
    data_path = Path(path) if path else DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"SVAMP data not found at {data_path}. Download SVAMP.json from "
            "https://github.com/arkilpatel/SVAMP into raps_data/svamp/."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        return svamp_data_process(json.load(f))
