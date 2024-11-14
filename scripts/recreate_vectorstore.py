import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings

from src.config import Config


def main() -> None:
    load_dotenv()
    config = Config()

    loader = DirectoryLoader(
        "",
        glob=["README.md", "docs/**/*.md"],
        recursive=True,
    )

    splits = loader.load_and_split()
    for doc in splits:
        doc.id = doc.metadata["source"]

    print(splits)  # noqa: T201

    client = chromadb.HttpClient(
        host=config.chroma_host.get_secret_value(),
        port=config.chroma_port.get_secret_value(),
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=config.chroma_token.get_secret_value(),
        ),
    )
    client.heartbeat()

    vectorstore = Chroma(
        "sample",
        OpenAIEmbeddings(api_key=config.openai_api_key),
        client=client,
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(splits)


if __name__ == "__main__":
    main()
