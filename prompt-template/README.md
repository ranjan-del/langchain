# Prompt Template

Build reusable, parameterized prompts with `{variable}` placeholders.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- Declaring a template with named placeholders and listing its input variables.
- Filling a template with `.format(**kwargs)`, including multi-variable prompts.
- Failing loudly with `KeyError` when a required variable is not supplied.
- Partial application: binding some variables now and the rest later.
- `ChatPromptTemplate.from_messages`, which renders a whole role-tagged
  conversation from one shared variable dict (a variable used in both the system
  and the human message is only supplied once).

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python prompt-template/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
