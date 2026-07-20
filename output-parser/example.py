"""Output parser - LangChain concept example (self-contained, offline).

Models return free-form text, but applications usually need structured data. An
output parser turns that raw text into a Python object (a list, a dict, a typed
record) and validates it.

This example implements two parsers that mirror LangChain's built-ins: a
comma-separated list parser and a simple ``key: value`` parser. It runs fully
offline on fixed sample text, so the parsed result is deterministic.
"""

from __future__ import annotations


class CommaSeparatedListParser:
    """Parse ``"a, b, c"`` into ``["a", "b", "c"]``."""

    def parse(self, text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]


class KeyValueParser:
    """Parse lines of ``key: value`` into a dict.

    Later duplicate keys overwrite earlier ones; lines without a colon are
    ignored. Values that look like integers are converted.
    """

    def parse(self, text: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            result[key] = int(value) if value.lstrip("-").isdigit() else value
        return result


def build_examples() -> dict[str, object]:
    """Run both parsers on sample model output and return the parsed objects."""
    list_text = "apples, bananas, cherries,  dates "
    record_text = "name: Ada Lovelace\nrole: mathematician\nborn: 1815\nnote"

    return {
        "list": CommaSeparatedListParser().parse(list_text),
        "record": KeyValueParser().parse(record_text),
    }


def main() -> None:
    """Entry point: parse sample text into structured objects and print them."""
    parsed = build_examples()
    print("[output-parser] structuring raw model text\n")
    print(f"  list parser   -> {parsed['list']}")
    print(f"  record parser -> {parsed['record']}")


if __name__ == "__main__":
    main()
