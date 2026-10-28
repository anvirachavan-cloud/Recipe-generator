"""
Document Ingestion & Vector Store Module
Handles loading, chunking, embedding, and storing culinary documents.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
COLLECTION_NAME = "culinary_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Initialize HuggingFace sentence-transformer embeddings."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(documents_dir: str = DOCUMENTS_DIR) -> List[Document]:
    """Load all supported documents from the documents directory."""
    docs: List[Document] = []
    doc_path = Path(documents_dir)

    if not doc_path.exists():
        logger.warning("Documents directory not found: %s", documents_dir)
        return docs

    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.txt": TextLoader,
        "**/*.docx": Docx2txtLoader,
    }

    for glob_pattern, loader_cls in loaders.items():
        for file_path in doc_path.glob(glob_pattern):
            try:
                logger.info("Loading document: %s", file_path)
                if loader_cls == TextLoader:
                    loader = loader_cls(str(file_path), encoding="utf-8")
                else:
                    loader = loader_cls(str(file_path))
                loaded = loader.load()
                # Tag each document with its source category
                for doc in loaded:
                    doc.metadata["category"] = infer_category(file_path.name)
                docs.extend(loaded)
                logger.info("Loaded %d pages from %s", len(loaded), file_path.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", file_path, exc)

    logger.info("Total documents loaded: %d", len(docs))
    return docs


def infer_category(filename: str) -> str:
    """Infer document category from filename."""
    name = filename.lower()
    if any(k in name for k in ["cookbook", "recipe", "cooking"]):
        return "cookbook"
    if any(k in name for k in ["diet", "nutrition", "guideline", "health"]):
        return "dietary_guideline"
    if any(k in name for k in ["allergy", "allergen", "safety", "food_safety"]):
        return "food_safety"
    return "general_culinary"


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def build_vector_store(
    chunks: List[Document],
    persist_dir: str = CHROMA_PERSIST_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    """Build or load the ChromaDB vector store."""
    embeddings = get_embeddings()
    persist_path = Path(persist_dir)

    if persist_path.exists() and not force_rebuild:
        logger.info("Loading existing vector store from %s", persist_dir)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        count = vectorstore._collection.count()
        if count > 0:
            logger.info("Vector store loaded with %d vectors.", count)
            return vectorstore
        logger.info("Vector store empty — rebuilding.")

    if not chunks:
        logger.warning("No document chunks provided. Vector store will be empty.")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        return vectorstore

    logger.info("Building vector store with %d chunks...", len(chunks))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    logger.info("Vector store built and persisted at %s", persist_dir)
    return vectorstore


def get_or_create_vector_store(force_rebuild: bool = False) -> Chroma:
    """High-level helper: load documents, chunk them, and return a vector store."""
    documents = load_documents()
    chunks = chunk_documents(documents) if documents else []
    return build_vector_store(chunks, force_rebuild=force_rebuild)


def add_documents_to_store(
    vectorstore: Chroma, new_docs: List[Document]
) -> Chroma:
    """Incrementally add new documents to an existing vector store."""
    if not new_docs:
        return vectorstore
    chunks = chunk_documents(new_docs)
    vectorstore.add_documents(chunks)
    logger.info("Added %d new chunks to vector store.", len(chunks))
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 5):
    """Return a similarity-search retriever."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def search_by_category(
    vectorstore: Chroma, query: str, category: str, k: int = 3
) -> List[Document]:
    """Filter retrieval by document category metadata."""
    results = vectorstore.similarity_search(
        query,
        k=k * 3,  # over-fetch then filter
        filter={"category": category},
    )
    return results[:k]
