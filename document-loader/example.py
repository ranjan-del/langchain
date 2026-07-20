"""Document loader - LangChain concept example (self-contained, offline).

A document loader reads a source (a file, a URL, a database row) and returns a
list of ``Document`` objects. Each document holds ``page_content`` (the text) and
``metadata`` (where it came from). This is the first stage of most retrieval
pipelines.

This example implements a ``Document`` record and a plain-text loader that mirror
``langchain_core.documents.Document`` and the community ``TextLoader``. It writes
a small sample file to a temp directory, loads it, and reports the documents. It
runs offline with no external dependencies.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class Document:
    """A unit of loaded text plus its provenance metadata."""

    page_content: str
    metadata: dict[str, object] = field(default_factory=dict)


class TextLoader:
    """Load a UTF-8 text file into a single :class:`Document`."""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> list[Document]:
        with open(self.path, "r", encoding="utf-8") as handle:
            text = handle.read()
        metadata = {"source": self.path, "chars": len(text)}
        return [Document(page_content=text, metadata=metadata)]


class TextDirectoryLoader:
    """Load every ``.txt`` file in a directory, one :class:`Document` each."""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def load(self) -> list[Document]:
        docs: list[Document] = []
        for name in sorted(os.listdir(self.directory)):
            if name.endswith(".txt"):
                docs.extend(TextLoader(os.path.join(self.directory, name)).load())
        return docs


def load_sample_documents() -> list[Document]:
    """Create sample text files in a temp dir and load them into documents."""
    tmp = tempfile.mkdtemp(prefix="doc_loader_")
    samples = {
        "intro.txt": "LangChain loads documents from many sources.",
        "usage.txt": "Each document keeps its page_content and metadata.",
    }
    for name, content in samples.items():
        with open(os.path.join(tmp, name), "w", encoding="utf-8") as handle:
            handle.write(content)
    return TextDirectoryLoader(tmp).load()


def main() -> None:
    """Entry point: load sample documents and print their content + metadata."""
    docs = load_sample_documents()
    print(f"[document-loader] loaded {len(docs)} document(s)\n")
    for i, doc in enumerate(docs, start=1):
        print(f"  [{i}] content : {doc.page_content}")
        print(f"      metadata: {doc.metadata}\n")


if __name__ == "__main__":
    main()
