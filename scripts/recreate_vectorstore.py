import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings


def main() -> None:
    load_dotenv()

    loader = DirectoryLoader(
        "",
        glob=["README.md", "docs/**/*.md"],
        recursive=True,
    )

    splits = loader.load_and_split()
    for doc in splits:
        doc.id = doc.metadata["source"]

    print(splits)  # noqa: T201

    chroma_host = os.getenv("CHROMA_HOST")
    assert chroma_host is not None  # noqa: S101

    client = chromadb.HttpClient(
        host=chroma_host,
        port=8000,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=os.getenv("CHROMA_TOKEN"),
        ),
    )
    client.heartbeat()

    vectorstore = Chroma("sample", OpenAIEmbeddings(), client=client)
    vectorstore.reset_collection()
    vectorstore.add_documents(splits)


if __name__ == "__main__":
    main()
