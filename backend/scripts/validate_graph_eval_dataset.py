"""Read-only reference validation for a graph-sensitive evaluation asset."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import Settings
from app.deps import build_container
from app.evaluation.dataset import (
    load_evaluation_dataset,
    validate_dataset_references,
)

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_validator_settings(env_file: Path = PROJECT_ENV_FILE) -> Settings:
    """Load the deployment env explicitly instead of depending on shell cwd."""
    return Settings(_env_file=env_file)


async def validate(path: Path) -> tuple[int, int, int, int]:
    dataset = load_evaluation_dataset(path)
    settings = load_validator_settings()
    container = build_container(settings)
    if container.mongo is None:
        raise RuntimeError("MongoDB unavailable")
    await container.mongo.connect()
    try:
        if not await container.graph_store.connect():
            raise RuntimeError("Neo4j unavailable")
        active_chunk_ids = {str(chunk["_id"]) for chunk in await container.store.list_active_chunks()}
        async with container.graph_store._driver.session(database=settings.neo4j_database) as session:
            entity_result = await session.run("MATCH (entity:Entity) RETURN entity.key AS key")
            relationship_result = await session.run("MATCH ()-[relationship:RELATES_TO]->() RETURN relationship.key AS key")
            entity_keys = {str(row["key"]) async for row in entity_result}
            relationship_keys = {str(row["key"]) async for row in relationship_result}
        errors = validate_dataset_references(dataset, active_chunk_ids, entity_keys, relationship_keys)
        return len(dataset.cases), sum(case.graph_sensitive for case in dataset.cases), len(errors), len(active_chunk_ids)
    finally:
        await container.mongo.close()
        await container.graph_store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        cases, graph_cases, error_count, active_chunks = asyncio.run(validate(args.dataset))
    except RuntimeError as exc:
        print(f"validation_status=unavailable reason={exc}")
        raise SystemExit(2) from exc
    print(f"cases={cases} graph_sensitive_cases={graph_cases} active_chunks={active_chunks} errors={error_count}")
    raise SystemExit(1 if error_count else 0)


if __name__ == "__main__":
    main()
