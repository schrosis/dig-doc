from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStore as BaseVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

from src.config import ChromaConfig


class VectorStore:
    _store: BaseVectorStore

    def __init__(self, store: BaseVectorStore) -> None:
        self._store = store

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:  # noqa: ANN401
        return self._store.as_retriever(**kwargs)

    def add_documents(self, documents: list[Document]) -> None:
        self._store.add_documents(documents)

    def update_documents(self, ids: list[str], documents: list[Document]) -> None:
        self._store.add_documents(documents, ids=ids)

    def delete_documents(self, ids: list[str]) -> None:
        self._store.delete(ids)

    def delete_all_documents(self) -> None:
        if isinstance(self._store, Chroma):
            # chroma は delete() を正しく実装できていないため
            # 代わりに reset_collection() を呼び出す
            self._store.reset_collection()
        else:
            self._store.delete()


class VectorStoreFactory:
    _client: chromadb.ClientAPI | None = None

    def __init__(self, config: ChromaConfig) -> None:
        self.config = config

    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self.config.host.get_secret_value(),
                port=self.config.port.get_secret_value(),
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                    chroma_client_auth_credentials=self.config.token.get_secret_value(),
                ),
            )
            self._client.heartbeat()

        return self._client

    def create(self, collection_name: str, embeddings: Embeddings) -> VectorStore:
        chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=self.client(),
        )

        return VectorStore(chroma)
