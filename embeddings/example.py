"""Embeddings - LangChain concept example (self-contained, offline).

An embedding maps text to a fixed-length vector of numbers so that semantically
similar texts land near each other. Similarity is usually measured with cosine
similarity. Hosted embedding models (OpenAI, etc.) give the best vectors but need
an API key.

To stay offline and deterministic, this example uses a hashing embedder: it
buckets word hashes into a fixed-size, L2-normalized vector. The vectors are not
state-of-the-art, but they are reproducible and demonstrate the full workflow
(embed -> store -> search) with zero external dependencies. The interface
(``embed(text) -> vector``) matches LangChain's ``Embeddings`` base class.

It also includes a minimal ``InMemoryVectorStore``. Embedding a text is only half
the job: production retrieval needs the vectors kept alongside their documents so
a query can be scored against all of them at once and the top ``k`` returned. That
store is the piece that turns an embedder into retrieval.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field


class HashingEmbedder:
    """Deterministic, offline embedder based on hashed word buckets."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Return an L2-normalized vector of length ``self.dim`` for ``text``."""
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed(text) for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def most_similar(query: str, documents: list[str], embedder: HashingEmbedder) -> tuple[str, float]:
    """Return the document most similar to ``query`` and its score."""
    query_vec = embedder.embed(query)
    scored = [
        (doc, cosine_similarity(query_vec, embedder.embed(doc))) for doc in documents
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0]


@dataclass
class InMemoryVectorStore:
    """Hold documents next to their vectors and answer top-k queries.

    Embedding is deliberately done once at insert time and never again. The whole
    reason a vector store exists is that re-embedding the corpus for every query
    is the expensive part; scoring pre-computed vectors is cheap.

    The scan is linear, which is correct and fast enough for a handful of
    documents. Real stores add an approximate-nearest-neighbour index because
    linear scan stops being viable somewhere in the millions.
    """

    embedder: HashingEmbedder
    texts: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add_texts(self, texts: list[str]) -> None:
        """Embed and store a batch of documents."""
        for text in texts:
            self.texts.append(text)
            self.vectors.append(self.embedder.embed(text))

    def similarity_search(self, query: str, k: int = 2) -> list[tuple[str, float]]:
        """Return the ``k`` closest documents as ``(text, score)``, best first."""
        query_vec = self.embedder.embed(query)
        scored = [
            (text, cosine_similarity(query_vec, vector))
            for text, vector in zip(self.texts, self.vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def main() -> None:
    """Entry point: embed sentences, rank them against a query, print scores."""
    embedder = HashingEmbedder(dim=128)
    documents = [
        "Cats and dogs are common household pets.",
        "The stock market rallied on strong earnings.",
        "Kittens and puppies are baby pets that people love.",
    ]
    query = "pets like cats and dogs"

    print("[embeddings] ranking documents by cosine similarity\n")
    query_vec = embedder.embed(query)
    for doc in documents:
        score = cosine_similarity(query_vec, embedder.embed(doc))
        print(f"  {score:.3f}  {doc}")

    best, best_score = most_similar(query, documents, embedder)
    print(f"\n  query      : {query}")
    print(f"  best match : {best} (score {best_score:.3f})")

    # A vector store embeds once at insert time, then answers many queries.
    store = InMemoryVectorStore(embedder)
    store.add_texts(documents)
    print(f"\n  vector store holds {len(store.texts)} document(s)")
    print("  similarity_search(query, k=2):")
    for rank, (text, score) in enumerate(store.similarity_search(query, k=2), start=1):
        print(f"    {rank}. {score:.3f}  {text}")


if __name__ == "__main__":
    main()
