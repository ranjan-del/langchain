# Embeddings

Turn text into vectors and rank documents by cosine similarity.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- A deterministic, offline hashing embedder producing L2-normalized vectors.
- Cosine similarity between a query and candidate documents.
- Ranking documents and selecting the most similar one.
- An `InMemoryVectorStore` with `add_texts` and `similarity_search(query, k)`,
  which embeds each document once at insert time and then scores many queries
  against the stored vectors. Re-embedding the corpus per query is the expensive
  part, and avoiding it is the entire reason vector stores exist.
- The honest limit of the approach: the hashing embedder matches shared words,
  not meaning, so the paraphrase about kittens and puppies scores lower (0.405)
  than the literal match (0.676).

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python embeddings/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
