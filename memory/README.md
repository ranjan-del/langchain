# Memory

Persist conversational context so the model appears to remember earlier turns.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- A `ConversationBufferMemory` that records every user and AI turn.
- Replaying the buffered transcript on each turn.
- A model recalling a fact stated earlier, entirely offline.
- Three competing strategies run against the identical 4-turn script:
  `ConversationBufferMemory` (keep everything), `ConversationBufferWindowMemory`
  (keep the last k turns) and `ConversationSummaryMemory` (a lossy summary plus
  the last k turns).
- The trade-off made concrete: only the full buffer still answers "what is my
  name?" correctly. The window dropped the introduction and the summary did not
  preserve it. Bounding the prompt size costs you recall.

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python memory/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
