# Text Splitter

Split long text into overlapping chunks for embedding or context windows.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- A character splitter with `chunk_size` and `chunk_overlap` settings.
- Sliding an overlapping window across the text and tracking each chunk's span.
- Validation that overlap must be smaller than chunk size.
- A `RecursiveCharacterTextSplitter` (LangChain's default) that tries separators
  from coarsest to finest (paragraph, newline, sentence, space, character) and
  only hard-cuts as a last resort.
- Why the recursive one is preferred: run the example and compare. The blind
  window ends chunk 1 at `"Overlap keeps context f"`, mid-word. The recursive
  splitter ends its chunks at sentence boundaries, which embeds better and reads
  better when a chunk is shown to a user as a citation.
- `chunk_size` treated as a hard bound: the carried-over overlap is dropped
  rather than allowed to push a chunk past the limit.

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python text-splitter/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
