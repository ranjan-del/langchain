"""Embeddings - LangChain concept example (self-contained, offline).

An embedding maps text to a fixed-length vector of numbers so that semantically
similar texts land near each other. Similarity is usually measured with cosine
similarity. Hosted embedding models (OpenAI, etc.) give the best vectors but need
an API key.

To stay offline and deterministic, this example uses a hashing embedder: it
buckets word hashes into a fixed-size, L2-normalized vector. The vectors are not
state-of-the-art, but they are reproducible and demonstrate the full workflow
(embed -> compare -> rank) with zero external dependencies. The interface
(``embed(text) -> vector``) matches LangChain's ``Embeddings`` base class.
"""

from __future__ import annotations

import hashlib
import math


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


if __name__ == "__main__":
    main()
