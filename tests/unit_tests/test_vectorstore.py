
from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore as BaseVectorStore

from src.vectorstore import VectorStore


@pytest.fixture
def mock_base_vector_store() -> MagicMock:
    return MagicMock(spec=BaseVectorStore)

def test_add_documents(mock_base_vector_store: MagicMock) -> None:
    vector_store = VectorStore(mock_base_vector_store)
    documents = [
        Document(
            page_content="Test content",
            metadata={"source": "test_source", "id": "test_id"},
        ),
    ]

    vector_store.add_documents(documents)

    mock_base_vector_store.add_documents.assert_called_once_with(documents)

def test_update_documents(mock_base_vector_store: MagicMock) -> None:
    vector_store = VectorStore(mock_base_vector_store)
    documents = [
        Document(
            page_content="Updated content",
            metadata={"source": "test_source", "id": "test_id"},
        ),
    ]
    ids = ["test_id"]

    vector_store.update_documents(ids, documents)

    mock_base_vector_store.add_documents.assert_called_once_with(documents, ids=ids)

def test_delete_documents(mock_base_vector_store: MagicMock) -> None:
    vector_store = VectorStore(mock_base_vector_store)
    ids = ["test_id1", "test_id2"]

    vector_store.delete_documents(ids)

    mock_base_vector_store.delete.assert_called_once_with(ids)

def test_delete_all_documents_chroma(mock_base_vector_store: MagicMock) -> None:
    mock_base_vector_store.reset_collection = MagicMock()
    mock_base_vector_store.mock_add_spec(Chroma)
    vector_store = VectorStore(mock_base_vector_store)

    vector_store.delete_all_documents()

    mock_base_vector_store.reset_collection.assert_called_once()

def test_delete_all_documents_other(mock_base_vector_store: MagicMock) -> None:
    vector_store = VectorStore(mock_base_vector_store)

    vector_store.delete_all_documents()

    mock_base_vector_store.delete.assert_called_once()
