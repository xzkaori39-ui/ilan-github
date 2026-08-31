"""Import a validated i兰 GraphRAG JSON projection into Neo4j."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.graph.store import GraphStore


async def run(projection_path: Path) -> dict:
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    settings = get_settings()
    graph = GraphStore(
        enabled=True,
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        imported = await graph.import_graphrag_projection(projection)
        return {"imported": imported, "graph": await graph.summary()}
    finally:
        await graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.projection)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
