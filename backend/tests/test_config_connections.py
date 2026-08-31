from pathlib import Path

from app.config import Settings


def test_placeholder_database_urls_are_repaired_from_shared_credentials(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "MONGO_INITDB_ROOT_USERNAME=wenshu_admin",
                "MONGO_INITDB_ROOT_PASSWORD=real-mongo-password",
                "MONGODB_URI=mongodb://wenshu_admin:change-me-strong-mongo-password@localhost:27017/wenshu?authSource=admin",
                "REDIS_PASSWORD=real-redis-password",
                "REDIS_ADDR=redis://:change-me-strong-redis-password@localhost:6379",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env)

    assert settings.mongodb_uri == (
        "mongodb://wenshu_admin:real-mongo-password@localhost:27017/wenshu?authSource=admin"
    )
    assert settings.redis_addr == "redis://:real-redis-password@localhost:6379"


def test_explicit_real_urls_are_not_overwritten(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "MONGO_INITDB_ROOT_USERNAME=wenshu_admin",
                "MONGO_INITDB_ROOT_PASSWORD=other-password",
                "MONGODB_URI=mongodb://service:real-password@db.internal:27017/app?authSource=admin",
                "REDIS_PASSWORD=other-password",
                "REDIS_ADDR=redis://:real-password@redis.internal:6379/2",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env)

    assert settings.mongodb_uri == "mongodb://service:real-password@db.internal:27017/app?authSource=admin"
    assert settings.redis_addr == "redis://:real-password@redis.internal:6379/2"
