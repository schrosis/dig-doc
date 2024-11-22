import os

from pydantic import Secret, SecretStr


class ChromaConfig:
    host: SecretStr
    port: Secret[int]
    token: SecretStr

    def __init__(self) -> None:
        host = os.getenv("CHROMA_HOST")
        if host is None or host == "":
            raise ValueError("environment variable CHROMA_HOST is not set")
        self.host = SecretStr(host)

        port = os.getenv("CHROMA_PORT")
        if port is None or port == "" or not port.isdigit():
            raise ValueError("environment variable CHROMA_PORT is not set")
        self.port = Secret(int(port))

        token = os.getenv("CHROMA_TOKEN")
        if token is None or token == "":
            raise ValueError("environment variable CHROMA_TOKEN is not set")
        self.token = SecretStr(token)


class OpenAIConfig:
    api_key: SecretStr

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None or api_key == "":
            raise ValueError("environment variable OPENAI_API_KEY is not set")
        self.api_key = SecretStr(api_key)


class Config:
    openai_config: OpenAIConfig
    chroma_config: ChromaConfig

    def __init__(self) -> None:
        self.openai_config = OpenAIConfig()
        self.chroma_config = ChromaConfig()
