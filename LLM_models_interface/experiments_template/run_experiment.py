"""
Run a failure-mode detection experiment driven by config.yaml.
Trace is loaded from data/splits/dev/.
"""

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, create_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_interface import judge


REPO_ROOT = Path(__file__).parent.parent.parent
TRACE = (REPO_ROOT / "data" / "splits" / "dev" / "trace_dummy.txt").read_text()

_TYPE_MAP = {"str": str, "bool": bool, "int": int, "float": float}


def build_schema(schema_cfg: dict) -> type[BaseModel]:
    fields: dict[str, Any] = {
        f["name"]: (_TYPE_MAP[f["type"]], ...)
        for f in schema_cfg["fields"]
    }
    return create_model("DetectionResult", **fields)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: Path = Path(__file__).parent / "config.yaml"):
    config = load_config(config_path)
    exp = config["experiment"]

    model = exp["model"]
    system_prompt = exp["system_prompt"]
    schema = build_schema(exp["schema"])
    user_prompt = exp["user_prompt"].format(trace=TRACE)

    print(f"Experiment : {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Model      : {model}\n")

    resp = judge(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
    )

    parsed_fields = resp.parsed.model_dump() if resp.parsed else {"raw_text": resp.raw_text}
    for key, val in parsed_fields.items():
        print(f"{key.capitalize():<12}: {val}")
    print(f"Cost        : ${resp.cost_usd:.6f}  |  Latency: {resp.latency_s:.2f}s")

    output_path = Path(__file__).parent / "results.jsonl"
    record = {
        "model": model,
        **parsed_fields,
        "tokens_in": resp.tokens_in,
        "tokens_out": resp.tokens_out,
        "cost_usd": resp.cost_usd,
        "latency_s": resp.latency_s,
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nResult appended to {output_path}")


if __name__ == "__main__":
    run()
