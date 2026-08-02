"""Agents - LangChain concept example (self-contained, offline).

An agent uses a model to decide which tool to call for a given request, runs the
tool, and returns the result. Real agents let an LLM do the routing; that needs
an API key.

To stay offline and deterministic, this example uses a small rule-based router
as the "reasoning" step. It inspects the query, selects a tool from a registry,
executes it, and reports the reasoning trace. The structure (tools + a router +
an execution loop) matches how a LangChain agent is wired.

Two things separate an agent from a plain chain, and both are shown here:

* the tool is *chosen* at runtime rather than fixed when the pipeline is built,
  which also means the router must be allowed to choose nothing at all;
* the loop is *iterative* - each observation is written to a scratchpad and fed
  into the next decision, so several tools can be combined for one request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """A named callable the agent can choose to run."""

    name: str
    func: Callable[[str], str]


def _calculator(query: str) -> str:
    """Evaluate a simple two-operand expression found in the query.

    Deliberately a regex over the four basic operators rather than ``eval``:
    an agent tool runs whatever the model asked for, so handing it an arbitrary
    Python evaluator is a genuine remote-code-execution hole.
    """
    match = re.search(r"(-?\d+)\s*([\+\-\*/])\s*(-?\d+)", query)
    if not match:
        return "no arithmetic expression found"
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    if op == "/" and b == 0:
        return "division by zero"
    value = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
    # Keep integer results looking like integers (4/2 should read as 2, not 2.0).
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _length(query: str) -> str:
    """Return the number of characters in the quoted phrase, else in the query."""
    quoted = re.search(r"'([^']*)'", query)
    target = quoted.group(1) if quoted else query
    return str(len(target))


def _reverse(query: str) -> str:
    """Reverse the quoted phrase in the query."""
    quoted = re.search(r"'([^']*)'", query)
    target = quoted.group(1) if quoted else query
    return target[::-1]


TOOLS = {
    "calculator": Tool("calculator", _calculator),
    "length": Tool("length", _length),
    "reverse": Tool("reverse", _reverse),
}


ARITHMETIC = re.compile(r"-?\d+\s*[\+\-\*/]\s*-?\d+")


def route(query: str) -> str | None:
    """Rule-based reasoning: pick the tool best matching the query, or none.

    Explicit intent words are tested before the arithmetic pattern so that a
    request like "reverse '3+4'" is not hijacked by the calculator.

    Returns ``None`` when nothing matches. An earlier version fell through to the
    ``length`` tool, which meant the agent silently answered every unrecognised
    question with a character count. Admitting there is no matching tool is the
    correct behaviour and lets the caller decide what to do next.
    """
    lowered = query.lower()
    if "reverse" in lowered or "backwards" in lowered:
        return "reverse"
    if "how long" in lowered or "length" in lowered or "how many characters" in lowered:
        return "length"
    if ARITHMETIC.search(query):
        return "calculator"
    return None


def run_agent(query: str) -> dict[str, str]:
    """Route the query to a tool, run it, and return a trace of the decision."""
    tool_name = route(query)
    if tool_name is None:
        return {
            "query": query,
            "chosen_tool": "none",
            "observation": "",
            "answer": "No tool matched this request.",
        }
    tool = TOOLS[tool_name]
    observation = tool.func(query)
    return {
        "query": query,
        "chosen_tool": tool_name,
        "observation": observation,
        "answer": f"The answer is {observation}.",
    }


@dataclass
class AgentStep:
    """One decide-act-observe cycle, as written to the scratchpad."""

    thought: str
    tool: str
    observation: str


def plan(query: str) -> list[str]:
    """Decide the ordered list of tools needed for a possibly multi-part query.

    A single ``route`` call can only answer a single-intent question. Requests
    like "reverse 'offline' and tell me how long it is" need two tools applied in
    sequence, so the planner returns them in the order they should run.
    """
    lowered = query.lower()
    steps: list[str] = []
    if "reverse" in lowered or "backwards" in lowered:
        steps.append("reverse")
    if "how long" in lowered or "length" in lowered or "how many characters" in lowered:
        steps.append("length")
    if not steps and ARITHMETIC.search(query):
        steps.append("calculator")
    return steps


def run_agent_loop(query: str, max_steps: int = 4) -> dict[str, object]:
    """Run tools iteratively, feeding each observation into the next step.

    The scratchpad is the whole point: step two operates on the *result* of step
    one, not on the original query. ``max_steps`` bounds the loop because a real
    agent driven by a model can otherwise cycle indefinitely.
    """
    steps: list[AgentStep] = []
    current = query
    for tool_name in plan(query)[:max_steps]:
        observation = TOOLS[tool_name].func(current)
        steps.append(
            AgentStep(
                thought=f"input is {current!r}, so use the {tool_name} tool",
                tool=tool_name,
                observation=observation,
            )
        )
        # Quote the observation so the next tool's regexes see it as the target.
        current = f"'{observation}'"

    if not steps:
        return {"query": query, "steps": [], "answer": "No tool matched this request."}
    return {
        "query": query,
        "steps": steps,
        "answer": f"The answer is {steps[-1].observation}.",
    }


def main() -> None:
    """Entry point: run the agent on a few queries and print the traces."""
    print("[agents] routing queries to tools\n")
    queries = [
        "What is 6 * 7?",
        "What is 100 - 58?",
        "How long is the phrase 'langchain'?",
        "Please reverse 'offline'.",
        "What is the capital of France?",
    ]
    for query in queries:
        trace = run_agent(query)
        print(f"  query        : {trace['query']}")
        print(f"  chosen tool  : {trace['chosen_tool']}")
        print(f"  observation  : {trace['observation']}")
        print(f"  answer       : {trace['answer']}\n")

    # Multi-step: the second tool consumes the first tool's observation.
    multi = "Reverse 'langchain' and tell me how long the result is."
    result = run_agent_loop(multi)
    print("  multi-step loop")
    print(f"  query        : {result['query']}")
    for i, step in enumerate(result["steps"], start=1):  # type: ignore[arg-type]
        print(f"    step {i}: thought      : {step.thought}")
        print(f"            tool         : {step.tool}")
        print(f"            observation  : {step.observation}")
    print(f"  answer       : {result['answer']}")


if __name__ == "__main__":
    main()
