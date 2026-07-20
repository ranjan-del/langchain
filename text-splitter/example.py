"""Text splitter - LangChain concept example (self-contained, offline).

Long documents must be broken into smaller overlapping chunks before they can be
embedded or fed to a model with a limited context window. A text splitter does
this: it slides a fixed-size window across the text with a configurable overlap
so adjacent chunks share context.

This example implements a character splitter with the same ``chunk_size`` /
``chunk_overlap`` semantics as LangChain's ``CharacterTextSplitter``. It runs
offline and is fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A slice of the source text plus where it came from."""

    text: str
    start: int
    end: int


class CharacterTextSplitter:
    """Split text into overlapping fixed-size character windows."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        """Return the ordered list of overlapping chunks covering ``text``."""
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(text=text[start:end], start=start, end=end))
            if end == len(text):
                break
            start += step
        return chunks


SAMPLE = (
    "LangChain splits long documents into overlapping chunks. "
    "Overlap keeps context from spilling across boundaries. "
    "Each chunk can then be embedded and retrieved independently. "
    "Smaller chunks fit inside a model's context window."
)


def split_sample(chunk_size: int = 80, chunk_overlap: int = 20) -> list[Chunk]:
    """Split the built-in sample text and return the chunks."""
    return CharacterTextSplitter(chunk_size, chunk_overlap).split(SAMPLE)


def main() -> None:
    """Entry point: split the sample text and print each chunk with its span."""
    chunks = split_sample()
    print(f"[text-splitter] {len(chunks)} chunk(s) from {len(SAMPLE)} chars\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] ({chunk.start:>3}-{chunk.end:>3}) {chunk.text!r}")


if __name__ == "__main__":
    main()
