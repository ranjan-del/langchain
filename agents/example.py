"""Agents - LangChain concept example (self-contained, offline).

An agent uses a model to decide which tool to call for a given request, runs the
tool, and returns the result. Real agents let an LLM do the routing; that needs
an API key.

To stay offline and deterministic, this example uses a small rule-based router
as the "reasoning" step. It inspects the query, selects a tool from a registry,
executes it, and reports the reasoning trace. The structure (tools + a router +
an execution loop) matches how a LangChain agent is wired.
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
    """Evaluate a simple 'a + b' or 'a * b' expression found in the query."""
    match = re.search(r"(\d+)\s*([\+\*])\s*(\d+)", query)
    if not match:
        return "no arithmetic expression found"
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    value = a + b if op == "+" else a * b
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


def route(query: str) -> str:
    """Rule-based reasoning: pick the tool name best matching the query."""
    lowered = query.lower()
    if re.search(r"\d+\s*[\+\*]\s*\d+", query):
        return "calculator"
    if "reverse" in lowered:
        return "reverse"
    if "how long" in lowered or "length" in lowered or "how many characters" in lowered:
        return "length"
    return "length"


def run_agent(query: str) -> dict[str, str]:
    """Route the query to a tool, run it, and return a trace of the decision."""
    tool_name = route(query)
    tool = TOOLS[tool_name]
    observation = tool.func(query)
    return {
        "query": query,
        "chosen_tool": tool_name,
        "observation": observation,
        "answer": f"The answer is {observation}.",
    }


def main() -> None:
    """Entry point: run the agent on a few queries and print the traces."""
    print("[agents] routing queries to tools\n")
    queries = [
        "What is 6 * 7?",
        "How long is the phrase 'langchain'?",
        "Please reverse 'offline'.",
    ]
    for query in queries:
        trace = run_agent(query)
        print(f"  query        : {trace['query']}")
        print(f"  chosen tool  : {trace['chosen_tool']}")
        print(f"  observation  : {trace['observation']}")
        print(f"  answer       : {trace['answer']}\n")


if __name__ == "__main__":
    main()
