# iLAN GitHub Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clean, self-contained iLAN repository that can be uploaded to GitHub without private corpus data or credentials.

**Architecture:** Copy only selected source and deployment assets into the designated empty release directory. Retain GraphRAG integration and a source-only Microsoft GraphRAG snapshot, while explicitly excluding all generated GraphRAG artifacts. Add one synthetic document and align the public documentation/configuration with the slim deployment.

**Tech Stack:** Python 3.11, FastAPI, Next.js 15, TypeScript, MongoDB, Redis, Neo4j, Docker Compose, Microsoft GraphRAG source snapshot.

**Spec:** `docs/superpowers/specs/2026-08-31-github-release-design.md`

## Global Constraints

- Release path is exactly `/home/ilan-github`; `/home/wenshu-project` is read-only source material.
- Never copy `.env`, `department_files/`, database data, generated dependencies, runtime logs, GraphRAG generated artifacts, runs, caches, backups, or historical private-corpus evaluation reports.
- Retain only source files from `third_party/microsoft-graphrag`; omit its generated/cache directories.
- Include exactly one synthetic Markdown knowledge-base example at `demo_data/campus_service_demo.md`.
- Public branding is i兰 / iLAN.

---

### Task 1: Create the source-only release tree

**Files:**
- Create: `backend/`, `web/`, `services/pi-agent/`, `deploy/`, `docs/`, `graphrag/`, `third_party/microsoft-graphrag/`, `docker-compose.yml`, `Makefile`, `.gitignore`

- [x] Copy tracked-like application source while excluding private data and generated directories.
- [x] Initialize `/home/ilan-github` as a Git repository with a public-safe `.gitignore`.
- [x] Verify no excluded directory or secret file exists in the release tree.

### Task 2: Add safe demonstration data and deployment path

**Files:**
- Create: `demo_data/campus_service_demo.md`
- Modify: `docker-compose.yml`
- Modify: `.env.example`

- [x] Add one fictional, author-created Markdown knowledge-base sample.
- [x] Mount the demo directory read-only for explicit import, without automatic ingestion.
- [x] Replace public configuration copy and Compose labels with i兰 / iLAN naming without changing service endpoints.

### Task 3: Publishable documentation

**Files:**
- Modify: `README.md`
- Modify: `graphrag/README.md`
- Modify: `backend/README.md`

- [x] Rewrite public setup instructions around Docker, separate LLM/embedding configuration, health checks, explicit demo import, GraphRAG construction, and data boundary.
- [x] Remove claims that private documents or their historical reports are included.

### Task 4: Validate the release artifact

**Files:**
- Verify: `backend/tests/`
- Verify: `web/`
- Verify: root `docker-compose.yml`

- [x] Run secret/private-data scans and inspect release size.
- [x] Run backend tests, web production build, and Docker Compose config validation.
- [x] Inspect `git status` and provide the exact initial-commit command without pushing remotely.
