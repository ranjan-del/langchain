"""Tools - LangChain concept example (self-contained, offline).

A tool is a plain function wrapped with a name, a description and a schema so a
model can decide to call it and pass arguments. LangChain exposes the ``@tool``
decorator for this.

This example implements a small ``@tool`` decorator that captures that metadata,
registers a few tools, derives a JSON schema from their type hints, and invokes
them. The schema matters: a model cannot call a tool it cannot see the signature
of, so the schema is literally the contract sent to the provider alongside the
prompt. Bad arguments raise ``ToolInvocationError`` rather than surfacing a raw
``TypeError``, because a model passing the wrong arguments is an expected event
that the agent loop should be able to catch and retry.

It runs offline with deterministic results.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable


class ToolInvocationError(ValueError):
    """Raised when a tool is called with missing or unknown arguments."""


# Map Python annotations onto JSON Schema type names, which is the vocabulary
# every model provider expects in a function/tool definition.
_JSON_TYPES: dict[object, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


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

    @property
    def args_schema(self) -> dict[str, object]:
        """Return a JSON Schema object describing this tool's parameters.

        Built from the function's own type hints so the schema cannot drift away
        from the implementation. Parameters without a default are required.
        """
        # eval_str resolves the annotations back into real types. This module
        # uses ``from __future__ import annotations``, so without it every
        # annotation would arrive as the string "int" and match nothing.
        signature = inspect.signature(self.func, eval_str=True)
        properties: dict[str, object] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            annotation = parameter.annotation
            properties[name] = {"type": _JSON_TYPES.get(annotation, "string")}
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        return {"type": "object", "properties": properties, "required": required}

    def invoke(self, **kwargs: object) -> object:
        """Call the underlying function with keyword arguments.

        Arguments are checked against the signature first so a wrong call from a
        model produces a clear, catchable error naming the offending fields.
        """
        expected = set(self.args)
        unknown = sorted(set(kwargs) - expected)
        if unknown:
            raise ToolInvocationError(f"{self.name}: unknown argument(s) {unknown}")
        required = set(self.args_schema["required"])  # type: ignore[arg-type]
        missing = sorted(required - set(kwargs))
        if missing:
            raise ToolInvocationError(f"{self.name}: missing argument(s) {missing}")
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


@tool
def repeat(text: str, times: int = 2) -> str:
    """Repeat text a number of times, joined by spaces."""
    return " ".join([text] * times)


def get_tools() -> list[Tool]:
    """Return the registered tools (the model's toolbox)."""
    return [add, word_count, repeat]


def main() -> None:
    """Entry point: list the tools, show their schemas, and invoke each one."""
    print("[tools] declaring and invoking callable tools\n")
    for t in get_tools():
        signature = ", ".join(t.args)
        print(f"  tool: {t.name}({signature}) - {t.description}")
    print()
    print(f"  add(a=2, b=40)                 -> {add.invoke(a=2, b=40)}")
    print(f"  word_count(text='a b c d e')   -> {word_count.invoke(text='a b c d e')}")
    print(f"  repeat(text='hi')              -> {repeat.invoke(text='hi')!r}")

    # This is the payload a provider receives so the model knows how to call it.
    print("\n  JSON schema advertised to the model:")
    for t in get_tools():
        print(f"    {t.name:<11} {t.args_schema}")

    # A model can and will call a tool wrongly; that must be catchable.
    try:
        add.invoke(a=1)
    except ToolInvocationError as exc:
        print(f"\n  bad call -> ToolInvocationError: {exc}")


if __name__ == "__main__":
    main()
