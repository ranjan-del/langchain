"""Text splitter - LangChain concept example (self-contained, offline).

Long documents must be broken into smaller overlapping chunks before they can be
embedded or fed to a model with a limited context window. A text splitter does
this: it slides a fixed-size window across the text with a configurable overlap
so adjacent chunks share context.

This example implements two splitters:

* ``CharacterTextSplitter``          - a blind fixed-size window, with the same
  ``chunk_size`` / ``chunk_overlap`` semantics as LangChain's class.
* ``RecursiveCharacterTextSplitter`` - LangChain's default and the one you should
  normally reach for. It tries a list of separators from coarsest to finest
  (paragraph, sentence, space, character) and only falls back to a hard cut when
  nothing else fits.

The second exists because the first slices words in half. A chunk ending mid-word
embeds badly and reads badly when it is shown to a user as a citation, so keeping
chunks on natural boundaries measurably improves retrieval quality.

Both run offline and are fully deterministic.
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


class RecursiveCharacterTextSplitter:
    """Split on the coarsest separator that produces small enough pieces.

    The separator list is ordered from most to least semantic. The splitter tries
    each in turn, and any piece still larger than ``chunk_size`` is re-split with
    the next separator down. The empty string is the last resort: it means "cut
    mid-word", which is exactly what we are trying to avoid but is still better
    than emitting an oversized chunk.
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively break ``text`` into pieces no larger than ``chunk_size``."""
        if len(text) <= self.chunk_size:
            return [text] if text else []
        if not separators:
            # Out of separators: hard-cut into fixed windows.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        if separator not in text:
            return self._split(text, rest)

        pieces: list[str] = []
        parts = text.split(separator)
        for index, part in enumerate(parts):
            # Put the separator back on every piece but the last, so that
            # rejoining the pieces reproduces the original text exactly.
            piece = part + separator if index < len(parts) - 1 else part
            if not piece:
                continue
            pieces.extend(self._split(piece, rest) if len(piece) > self.chunk_size else [piece])
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily pack small pieces back up to ``chunk_size``, with overlap.

        Splitting alone would leave a chunk per sentence, wasting the context
        window. Merging fills each chunk, then carries the tail of the previous
        chunk into the next one so a fact spanning a boundary is not lost.
        """
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self.chunk_size:
                chunks.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                # Carrying the tail must not itself push the new chunk over the
                # limit. Pieces are already guaranteed to fit on their own, so
                # when the overlap will not fit we drop it rather than emit an
                # oversized chunk. chunk_size is a hard bound, overlap is a
                # best-effort nicety.
                current = tail + piece if len(tail) + len(piece) <= self.chunk_size else piece
            else:
                current += piece
        if current:
            chunks.append(current)
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Return the ordered chunks for ``text``.

        Returns plain strings rather than :class:`Chunk` records: once pieces are
        merged and overlap is prepended, a chunk no longer maps to one contiguous
        span of the source, so reporting a start/end offset would be a lie.
        """
        if not text:
            return []
        return self._merge(self._split(text, self.separators))


SAMPLE = (
    "LangChain splits long documents into overlapping chunks. "
    "Overlap keeps context from spilling across boundaries. "
    "Each chunk can then be embedded and retrieved independently. "
    "Smaller chunks fit inside a model's context window."
)


def split_sample(chunk_size: int = 80, chunk_overlap: int = 20) -> list[Chunk]:
    """Split the built-in sample text and return the chunks."""
    return CharacterTextSplitter(chunk_size, chunk_overlap).split(SAMPLE)


def split_sample_recursively(chunk_size: int = 80, chunk_overlap: int = 20) -> list[str]:
    """Split the built-in sample on natural boundaries and return the chunks."""
    return RecursiveCharacterTextSplitter(chunk_size, chunk_overlap).split_text(SAMPLE)


def main() -> None:
    """Entry point: split the sample text and print each chunk with its span."""
    chunks = split_sample()
    print(f"[text-splitter] {len(chunks)} chunk(s) from {len(SAMPLE)} chars\n")
    print("  CharacterTextSplitter (blind fixed window):")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] ({chunk.start:>3}-{chunk.end:>3}) {chunk.text!r}")

    recursive = split_sample_recursively()
    print(f"\n  RecursiveCharacterTextSplitter ({len(recursive)} chunk(s), sentence-aware):")
    for i, text in enumerate(recursive, start=1):
        print(f"  [{i}] ({len(text):>3} chars) {text!r}")


if __name__ == "__main__":
    main()
