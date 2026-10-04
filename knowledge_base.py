import io
import re
from typing import Any

import numpy as np
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
BATCH_SIZE = 64  # the embeddings endpoint limits texts per request


class KnowledgeBase:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._client = None
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None  # unit-normalized rows

    # ------------------------------------------------------------------ setup
    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if self.api_key:
                self._client = genai.Client(api_key=self.api_key)
            else:
                self._client = genai.Client()
        return self._client

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings = None

    # ------------------------------------------------------------- extraction
    def extract_text(self, uploaded_file: Any) -> str:
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

    # --------------------------------------------------------------- indexing
    def add_document(self, filename: str, text: str) -> int:
        """Chunk, embed and index a document. Returns the number of chunks added."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return 0

        pieces = self._chunk_text(cleaned, size=900, overlap=120)

        # Embed first: if the API call fails, the existing index stays untouched.
        vectors = self._embed(pieces, task_type="RETRIEVAL_DOCUMENT")

        # Replace an existing upload with the same filename.
        self.remove_document(filename)

        self.chunks.extend({"source": filename, "text": p} for p in pieces)
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

        return len(pieces)

    def remove_document(self, filename: str) -> None:
        if not self.chunks:
            return

        keep = [i for i, c in enumerate(self.chunks) if c["source"] != filename]
        if len(keep) == len(self.chunks):
            return

        self.chunks = [self.chunks[i] for i in keep]
        if keep and self.embeddings is not None:
            self.embeddings = self.embeddings[keep]
        else:
            self.embeddings = None

    @staticmethod
    def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
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

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            response = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts[i : i + BATCH_SIZE],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(item.values for item in response.embeddings)

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks or self.embeddings is None:
            return []

        query_vector = self._embed([query], task_type="RETRIEVAL_QUERY")[0]
        scores = self.embeddings @ query_vector  # cosine similarity (both normalized)
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "source": self.chunks[i]["source"],
                "text": self.chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in indices
        ]