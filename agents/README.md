# Agents

Route a request to the right tool, run it, and report the reasoning trace.

This is a small, self-contained example. It is independent of the other folders in this repository and runs on its own with no API key and no external dependencies.

## What this demonstrates

- A registry of tools (calculator, length, reverse).
- A rule-based router that selects a tool for a query (offline stand-in for LLM reasoning).
- Executing the chosen tool and returning a trace of the decision.
- `route` returning `None` when nothing matches, so the agent says "no tool
  matched" instead of silently answering every unrecognised question.
- Explicit intent words winning over the arithmetic pattern, so `reverse '3+4'`
  is not hijacked by the calculator.
- `run_agent_loop`, a multi-step scratchpad loop where each observation becomes
  the input to the next tool, bounded by `max_steps`. This iterative decide,
  act, observe cycle is what separates an agent from a fixed chain.

## Prerequisites

- Python 3.10+
- No API key required. The example runs fully offline using a deterministic local implementation.
- Only the standard library is used at runtime; `pytest` (from the top-level `requirements.txt`) is needed to run the tests.

## How to run

```bash
# From the repository root:
python agents/example.py
```

## Notes

Where a production LangChain program would call a hosted model, this example uses a deterministic local stand-in so the result is reproducible and key-free. See the top-level README section "Implementation approach" for details.
