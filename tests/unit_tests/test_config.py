import os
from collections.abc import Generator
from typing import Never

import pytest
from pydantic import Secret, SecretStr

from src.config import Config


@pytest.fixture(autouse=True)
def set_env_vars() -> Generator[None, Never, None]:
    # テストの前に環境変数を設定する
    os.environ["OPENAI_API_KEY"] = "test_key"
    os.environ["CHROMA_HOST"] = "test_host"
    os.environ["CHROMA_PORT"] = "8000"
    os.environ["CHROMA_TOKEN"] = "test_token"
    yield
    # テストの後に環境変数を削除する
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "CHROMA_HOST" in os.environ:
        del os.environ["CHROMA_HOST"]
    if "CHROMA_PORT" in os.environ:
        del os.environ["CHROMA_PORT"]
    if "CHROMA_TOKEN" in os.environ:
        del os.environ["CHROMA_TOKEN"]

def test_config_initialization() -> None:
    config = Config()
    assert config.openai_config.api_key == SecretStr("test_key")
    assert config.chroma_config.host == SecretStr("test_host")
    assert config.chroma_config.port == Secret(8000)
    assert config.chroma_config.token == SecretStr("test_token")

@pytest.mark.parametrize(
    ("env_var"),
    [
        "OPENAI_API_KEY",
        "CHROMA_HOST",
        "CHROMA_PORT",
        "CHROMA_TOKEN",
    ],
)
def test_config_initialization_with_missing_env_vars(
    env_var: str,
) -> None:
    # 指定された環境変数を削除してテストする
    del os.environ[env_var]
    with pytest.raises(ValueError, match=f"environment variable {env_var} is not set"):
        Config()
