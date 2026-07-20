"""Tools - LangChain concept example (self-contained, offline).

A tool is a plain function wrapped with a name, a description and a schema so a
model can decide to call it and pass arguments. LangChain exposes the ``@tool``
decorator for this.

This example implements a small ``@tool`` decorator that captures that metadata,
registers a couple of tools (a calculator and a word counter), and invokes them
directly. It runs offline with deterministic results.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """A callable wrapped with metadata a model can inspect."""

    name: str
    description: str
    func: Callable[..., object]

    @property
    def args(self) -> list[str]:
        """Return the tool's parameter names (its call schema)."""
        return list(inspect.signature(self.func).parameters)

    def invoke(self, **kwargs: object) -> object:
        """Call the underlying function with keyword arguments."""
        return self.func(**kwargs)


def tool(func: Callable[..., object]) -> Tool:
    """Decorator: turn a function into a :class:`Tool`.

    The tool name is the function name and the description is its docstring's
    first line, matching how LangChain derives tool metadata.
    """
    doc = (func.__doc__ or "").strip().splitlines()
    description = doc[0] if doc else func.__name__
    return Tool(name=func.__name__, description=description, func=func)


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in text."""
    return len(text.split())


def get_tools() -> list[Tool]:
    """Return the registered tools (the model's toolbox)."""
    return [add, word_count]


def main() -> None:
    """Entry point: list the tools and invoke each one."""
    print("[tools] declaring and invoking callable tools\n")
    for t in get_tools():
        print(f"  tool: {t.name}{tuple(t.args)} - {t.description}")
    print()
    print(f"  add(a=2, b=40)                 -> {add.invoke(a=2, b=40)}")
    print(f"  word_count(text='a b c d e')   -> {word_count.invoke(text='a b c d e')}")


if __name__ == "__main__":
    main()
