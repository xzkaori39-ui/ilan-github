# iLAN GitHub Release Design

## Goal

Create a standalone, public GitHub release of iLAN from the working project without copying private campus documents, credentials, runtime databases, GraphRAG outputs, or internal evaluation reports.

## Release boundary

The release retains the complete application source: FastAPI backend, Next.js web client, pi-agent runtime, Docker Compose deployment, Neo4j/GraphRAG integration adapters, evaluation framework, tests, and a vendored Microsoft GraphRAG source snapshot.

It excludes all runtime and private material: `.env`, `department_files/`, GraphRAG inputs/outputs/caches/logs/runs/backups, generated front-end bundles, dependency directories, test caches, server logs, and historical reports tied to private corpus IDs. The vendored GraphRAG snapshot is source-only: no generated packages, caches, or build products.

## Demonstration knowledge base

The repository includes exactly one author-created Markdown file at `demo_data/campus_service_demo.md`. It uses no real university, person, policy, document, or account information. The document gives small fictional examples for course registration, grade review, and academic-status consultation. It is not ingested automatically; the README provides the explicit upload/import command after Docker startup.

## Documentation and configuration

The root README becomes the single public entry point under the i兰 / iLAN brand. It explains prerequisites, secrets configuration, Docker startup, demo import, health checks, GraphRAG scope, and public-data boundary. `.env.example` has no usable secrets, uses iLAN naming, and documents separate chat-model and embedding endpoints. Docker Compose and service labels use iLAN while preserving the existing service contracts.

## Verification

The release must pass a secret/private-data scan, contain no generated dependency or runtime directories, build the web client, pass backend tests, and validate Docker Compose syntax. No report that depends on the removed corpus may be represented as runnable in the public README.
