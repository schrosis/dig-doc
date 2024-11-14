import os

from pydantic import Secret, SecretStr


class Config:
    openai_api_key: SecretStr
    chroma_host: SecretStr
    chroma_port: Secret[int]
    chroma_token: SecretStr

    def __init__(self) -> None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None or openai_api_key == "":
            raise ValueError("environment variable OPENAI_API_KEY is not set")
        self.openai_api_key = SecretStr(openai_api_key)

        chroma_host = os.getenv("CHROMA_HOST")
        if chroma_host is None or chroma_host == "":
            raise ValueError("environment variable CHROMA_HOST is not set")
        self.chroma_host = SecretStr(chroma_host)

        chroma_port = os.getenv("CHROMA_PORT")
        if chroma_port is None or chroma_port == "" or not chroma_port.isdigit():
            raise ValueError("environment variable CHROMA_PORT is not set")
        self.chroma_port = Secret(int(chroma_port))

        chroma_token = os.getenv("CHROMA_TOKEN")
        if chroma_token is None or chroma_token == "":
            raise ValueError("environment variable CHROMA_TOKEN is not set")
        self.chroma_token = SecretStr(chroma_token)
