# Output Parser

Turn raw model text into structured Python objects.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- A comma-separated list parser producing a `list`.
- A `key: value` parser producing a `dict` with light type coercion.
- A `JsonOutputParser` that strips the ```` ```json ```` fence models habitually
  wrap their output in before calling `json.loads`.
- `get_format_instructions()` on every parser: the parser also tells the model
  what shape to answer in, and that text is pasted into the prompt.
- `OutputParserException` for unparseable or wrongly shaped output, so a caller
  can retry instead of receiving a silently wrong value.
- Running all three parsers on fixed sample text for a deterministic result.

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python output-parser/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
