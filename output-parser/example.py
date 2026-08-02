"""Output parser - LangChain concept example (self-contained, offline).

Models return free-form text, but applications usually need structured data. An
output parser turns that raw text into a Python object (a list, a dict, a typed
record) and validates it.

This example implements three parsers that mirror LangChain's built-ins: a
comma-separated list parser, a simple ``key: value`` parser, and a JSON parser
that tolerates the markdown code fence models habitually wrap JSON in.

Every parser also exposes ``get_format_instructions()``. That half is easy to
overlook but is the point of the abstraction: the parser both tells the model how
to shape its answer (the instructions are pasted into the prompt) and reads that
shape back. Parsing failures raise ``OutputParserException`` so a caller can
retry rather than crash on a silently wrong value.

It runs fully offline on fixed sample text, so the parsed result is deterministic.
"""

from __future__ import annotations

import json
import re


class OutputParserException(ValueError):
    """Raised when model output cannot be parsed into the requested shape.

    A dedicated type lets a caller catch exactly this failure and retry the model
    call, instead of confusing it with a bug in its own code.
    """


class CommaSeparatedListParser:
    """Parse ``"a, b, c"`` into ``["a", "b", "c"]``."""

    def get_format_instructions(self) -> str:
        """Return the instruction text to append to the prompt."""
        return "Respond with a comma-separated list, for example: foo, bar, baz"

    def parse(self, text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]


class JsonOutputParser:
    """Parse a JSON object out of model text, code fence and all.

    Models very often answer with ```json ... ``` even when told not to, so a
    parser that only calls ``json.loads`` fails on output that is actually
    correct. Stripping the fence first is the difference between a parser that
    works in practice and one that works in a demo.
    """

    _FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

    def get_format_instructions(self) -> str:
        """Return the instruction text to append to the prompt."""
        return "Respond with a single JSON object and nothing else."

    def parse(self, text: str) -> dict[str, object]:
        match = self._FENCE.search(text)
        payload = match.group(1) if match else text.strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OutputParserException(f"not valid JSON: {payload!r}") from exc
        if not isinstance(value, dict):
            raise OutputParserException(f"expected a JSON object, got {type(value).__name__}")
        return value


class KeyValueParser:
    """Parse lines of ``key: value`` into a dict.

    Later duplicate keys overwrite earlier ones; lines without a colon are
    ignored. Values that look like integers are converted.
    """

    def get_format_instructions(self) -> str:
        """Return the instruction text to append to the prompt."""
        return "Respond with one 'key: value' pair per line."

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
    # Deliberately fenced, which is how models usually return JSON.
    json_text = '```json\n{"name": "Ada Lovelace", "born": 1815}\n```'

    return {
        "list": CommaSeparatedListParser().parse(list_text),
        "record": KeyValueParser().parse(record_text),
        "json": JsonOutputParser().parse(json_text),
    }


def main() -> None:
    """Entry point: parse sample text into structured objects and print them."""
    parsed = build_examples()
    print("[output-parser] structuring raw model text\n")
    print(f"  list parser   -> {parsed['list']}")
    print(f"  record parser -> {parsed['record']}")
    print(f"  json parser   -> {parsed['json']}")

    print("\n  format instructions handed to the model:")
    for parser in (CommaSeparatedListParser(), KeyValueParser(), JsonOutputParser()):
        print(f"    {type(parser).__name__:<26} {parser.get_format_instructions()}")

    # Bad output must fail loudly and specifically, not return a wrong value.
    try:
        JsonOutputParser().parse("sorry, I could not answer that")
    except OutputParserException as exc:
        print(f"\n  invalid output -> OutputParserException: {exc}")


if __name__ == "__main__":
    main()
