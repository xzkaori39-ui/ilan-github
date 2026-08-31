"""Export active i兰 chunks to the official Microsoft GraphRAG JSONL input format."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.graph.graphrag_export import export_active_chunks
from app.storage.mongodb import MongoDB
from app.storage.store import build_store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "graphrag" / "input" / "active_chunks.jsonl"


async def run(output_path: Path) -> dict:
    settings = get_settings()
    mongo = MongoDB(settings)
    await mongo.connect()
    try:
        return await export_active_chunks(build_store(mongo), output_path)
    finally:
        await mongo.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.output)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
