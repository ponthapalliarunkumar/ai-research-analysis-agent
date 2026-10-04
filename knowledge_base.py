import io
import re
from typing import BinaryIO

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


class KnowledgeBase:
    def __init__(self):
        self.client = None
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings = None

    def extract_text(self, uploaded_file: BinaryIO) -> str:
        name = uploaded_file.name.lower()
        raw = uploaded_file.getvalue()

        if name.endswith(".txt"):
            return raw.decode("utf-8", errors="ignore")

        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if name.endswith(".docx"):
            document = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in document.paragraphs)

        raise ValueError("Unsupported file type.")

    def add_document(self, filename: str, text: str) -> None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return

        # Replace an existing upload with the same filename.
        self.chunks = [c for c in self.chunks if c["source"] != filename]
        self._rebuild_embeddings()

        pieces = self._chunk_text(cleaned, size=900, overlap=120)
        for piece in pieces:
            self.chunks.append({"source": filename, "text": piece})

        self._rebuild_embeddings()

    def _chunk_text(self, text: str, size: int, overlap: int) -> list[str]:
        if len(text) <= size:
            return [text]

        chunks = []
        start = 0
        step = max(1, size - overlap)

        while start < len(text):
            end = min(len(text), start + size)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start += step

        return chunks

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.client is None:
            self.client = genai.Client()

        model = "gemini-embedding-001"
        response = self.client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        vectors = [item.values for item in response.embeddings]
        return np.asarray(vectors, dtype=np.float32)

    def _rebuild_embeddings(self) -> None:
        if not self.chunks:
            self.embeddings = None
            return

        texts = [item["text"] for item in self.chunks]
        self.embeddings = self._embed(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks or self.embeddings is None:
            return []

        if self.client is None:
            self.client = genai.Client()

        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = np.asarray(response.embeddings[0].values, dtype=np.float32)

        matrix = self.embeddings
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_vector)

        scores = (matrix @ query_vector) / (
            np.maximum(matrix_norm[:, 0], 1e-12) * max(query_norm, 1e-12)
        )

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "source": self.chunks[i]["source"],
                "text": self.chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in indices
        ]
