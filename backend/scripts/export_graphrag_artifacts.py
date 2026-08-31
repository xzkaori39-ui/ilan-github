"""Convert official GraphRAG Parquet output into i兰's JSON graph projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.graph.graphrag_artifacts import build_projection


TABLES = ("text_units", "entities", "relationships", "communities", "community_reports")


def _records(path: Path) -> list[dict[str, Any]]:
    return json.loads(pd.read_parquet(path).to_json(orient="records", force_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="GraphRAG output directory containing Parquet tables")
    parser.add_argument("--output", required=True, type=Path, help="JSON projection path for the i兰 Neo4j importer")
    args = parser.parse_args()

    missing = [name for name in TABLES if not (args.input / f"{name}.parquet").is_file()]
    if missing:
        raise SystemExit(f"missing GraphRAG tables: {', '.join(missing)}")
    tables = {name: _records(args.input / f"{name}.parquet") for name in TABLES}
    projection = build_projection(**tables)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(projection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **{name: len(rows) for name, rows in projection.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
