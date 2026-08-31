"""The standalone graph-evaluation validator must not depend on the shell cwd."""
from __future__ import annotations


def test_validator_settings_load_the_explicit_project_env_file_from_any_working_directory(tmp_path, monkeypatch):
    from scripts.validate_graph_eval_dataset import load_validator_settings

    env_file = tmp_path / ".env"
    env_file.write_text("STORAGE_MODE=memory\nNEO4J_URI=bolt://example.test:7687\n", encoding="utf-8")
    other_cwd = tmp_path / "other-working-directory"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    settings = load_validator_settings(env_file)

    assert settings.storage_mode == "memory"
    assert settings.neo4j_uri == "bolt://example.test:7687"
