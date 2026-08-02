# Tools

Wrap plain functions as callable tools a model could choose to invoke.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- A `@tool` decorator that captures a name, description and argument schema.
- Inspecting a tool's metadata (name, args, description).
- `Tool.args_schema`, a JSON Schema derived from the function's own type hints so
  it cannot drift from the implementation. This is the contract actually sent to
  the model provider, and parameters with defaults are correctly optional.
- `ToolInvocationError` on missing or unknown arguments, because a model calling
  a tool wrongly is an expected event an agent loop should catch, not a crash.
- Invoking tools directly with keyword arguments.

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python tools/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
