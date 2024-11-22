from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings

from src.config import Config
from src.vectorstore import VectorStoreFactory


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

    vectorstore = VectorStoreFactory(config.chroma_config).create(
        "sample",
        embeddings=OpenAIEmbeddings(api_key=config.openai_config.api_key),
    )
    vectorstore.delete_all_documents()
    vectorstore.add_documents(splits)


if __name__ == "__main__":
    main()
