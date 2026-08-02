"""Document loader - LangChain concept example (self-contained, offline).

A document loader reads a source (a file, a URL, a database row) and returns a
list of ``Document`` objects. Each document holds ``page_content`` (the text) and
``metadata`` (where it came from). This is the first stage of most retrieval
pipelines.

This example implements a ``Document`` record and three loaders that mirror
``langchain_core.documents.Document``, the community ``TextLoader`` /
``DirectoryLoader`` and ``CSVLoader``. The CSV loader is included because it
shows the part that actually varies between loaders: not the reading, but the
decision of what counts as one document and what gets promoted to metadata. A
CSV yields one document per *row*, with the row number recorded so a later
retrieval step can cite its source.

It writes small sample files to a temp directory, loads them, and reports the
documents. It runs offline with no external dependencies.
"""

from __future__ import annotations

import csv
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


class CSVLoader:
    """Load a CSV file into one :class:`Document` per data row.

    ``page_content`` is the row rendered as ``column: value`` lines, which is the
    form a model reads most reliably. The row number goes into metadata rather
    than into the text so that retrieval can cite the exact row without the
    citation polluting the embedded content.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> list[Document]:
        docs: list[Document] = []
        with open(self.path, "r", encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle), start=1):
                content = "\n".join(f"{key}: {value}" for key, value in row.items())
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": self.path, "row": number},
                    )
                )
        return docs


def write_samples(directory: str) -> None:
    """Write the sample ``.txt`` and ``.csv`` fixtures into ``directory``."""
    samples = {
        "intro.txt": "LangChain loads documents from many sources.",
        "usage.txt": "Each document keeps its page_content and metadata.",
        "people.csv": "name,role\nAda Lovelace,mathematician\nAlan Turing,logician\n",
    }
    for name, content in samples.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(content)


def load_sample_documents() -> list[Document]:
    """Create sample text files in a temp dir and load them into documents."""
    tmp = tempfile.mkdtemp(prefix="doc_loader_")
    write_samples(tmp)
    # The directory loader only picks up ``.txt``, so the CSV fixture is ignored
    # here; ``load_sample_csv`` handles it with the loader built for that format.
    return TextDirectoryLoader(tmp).load()


def load_sample_csv() -> list[Document]:
    """Create a sample CSV in a temp dir and load one document per row.

    The temp directory is cleaned up before returning. The documents are already
    fully in memory at that point, so nothing is lost, and the example does not
    litter the system temp folder every time it runs.
    """
    with tempfile.TemporaryDirectory(prefix="doc_loader_csv_") as tmp:
        write_samples(tmp)
        return CSVLoader(os.path.join(tmp, "people.csv")).load()


def main() -> None:
    """Entry point: load sample documents and print their content + metadata."""
    docs = load_sample_documents()
    print(f"[document-loader] loaded {len(docs)} document(s)\n")
    for i, doc in enumerate(docs, start=1):
        print(f"  [{i}] content : {doc.page_content}")
        print(f"      metadata: {doc.metadata}\n")

    rows = load_sample_csv()
    print(f"  CSV loader: {len(rows)} document(s), one per row\n")
    for doc in rows:
        flattened = doc.page_content.replace("\n", " | ")
        print(f"  row {doc.metadata['row']}: {flattened}")


if __name__ == "__main__":
    main()
