"""Chains - LangChain concept example (self-contained, offline).

A chain composes small steps into a pipeline. LangChain's Expression Language
(LCEL) uses the ``|`` operator to pipe one step's output into the next, e.g.
``prompt | model | parser``.

This example implements a tiny ``Runnable`` with the same ``|`` behaviour and
builds the classic three-stage chain: a prompt template, a deterministic offline
model, and an output parser. No API key or network is needed, so the result is
reproducible.
"""

from __future__ import annotations

from typing import Any, Callable


class Runnable:
    """A single step in a pipeline.

    Wraps a function and supports ``a | b`` composition, where the output of
    ``a`` is fed into ``b``.
    """

    def __init__(self, func: Callable[[Any], Any], name: str = "step") -> None:
        self.func = func
        self.name = name

    def invoke(self, value: Any) -> Any:
        """Run this step on ``value``."""
        return self.func(value)

    def batch(self, values: list[Any]) -> list[Any]:
        """Run this step over many inputs.

        Every LCEL runnable exposes ``batch`` alongside ``invoke`` so a whole
        chain can be applied to a list without the caller rebuilding it each
        time. Inputs are independent, so order in equals order out.
        """
        return [self.invoke(value) for value in values]

    def __or__(self, other: "Runnable") -> "Runnable":
        """Compose two steps: ``(self | other).invoke(x) == other(self(x))``."""

        def piped(value: Any) -> Any:
            return other.invoke(self.invoke(value))

        return Runnable(piped, name=f"{self.name}|{other.name}")


def prompt_step(template: str) -> Runnable:
    """A step that fills ``{...}`` placeholders from a dict of variables."""
    return Runnable(lambda variables: template.format(**variables), name="prompt")


def model_step() -> Runnable:
    """A deterministic offline 'model' step.

    It stands in for a hosted LLM: given a prompt string it returns a
    reproducible response so the chain runs with no API key.
    """

    def fake_model(prompt: str) -> str:
        return f"MODEL_OUTPUT<{prompt.strip().lower()}>"

    return Runnable(fake_model, name="model")


def parser_step() -> Runnable:
    """A step that extracts the payload from the model output."""

    def parse(text: str) -> str:
        # Pull the content out of MODEL_OUTPUT<...> and title-case it.
        inner = text[len("MODEL_OUTPUT<"):-1] if text.startswith("MODEL_OUTPUT<") else text
        return inner.strip().title()

    return Runnable(parse, name="parser")


def parallel_step(branches: dict[str, Runnable]) -> Runnable:
    """Fan one input out to several named branches and collect a dict.

    This mirrors LangChain's ``RunnableParallel``. It matters because a chain is
    often not a straight line: you want the same input scored, summarised and
    classified, then merged into one structured payload for the next step.
    """

    def run_all(value: Any) -> dict[str, Any]:
        return {name: branch.invoke(value) for name, branch in branches.items()}

    return Runnable(run_all, name="parallel(" + ",".join(branches) + ")")


def build_chain() -> Runnable:
    """Compose prompt | model | parser into one runnable chain."""
    template = "Write a tagline about {topic} for {audience}."
    return prompt_step(template) | model_step() | parser_step()


def build_parallel_chain() -> Runnable:
    """Build ``prompt | model | {parsed, raw_length}``.

    The final stage runs two branches over the same model output, showing that
    composition is a graph, not only a straight pipe.
    """
    branches = {
        "parsed": parser_step(),
        "raw_length": Runnable(len, name="length"),
    }
    template = "Write a tagline about {topic} for {audience}."
    return prompt_step(template) | model_step() | parallel_step(branches)


def main() -> None:
    """Entry point: build the chain and invoke it end to end."""
    chain = build_chain()
    variables = {"topic": "offline testing", "audience": "developers"}
    result = chain.invoke(variables)
    print("[chains] prompt | model | parser\n")
    print(f"  chain steps : {chain.name}")
    print(f"  input       : {variables}")
    print(f"  result      : {result}")

    # The same chain applied to a list of inputs in one call.
    batch_inputs = [
        {"topic": "vector search", "audience": "students"},
        {"topic": "chunking", "audience": "librarians"},
    ]
    print("\n  batch:")
    for source, output in zip(batch_inputs, chain.batch(batch_inputs)):
        print(f"    {source['topic']:<14} -> {output}")

    # Fan-out: one model output consumed by two branches at once.
    parallel = build_parallel_chain()
    print(f"\n  parallel steps : {parallel.name}")
    print(f"  parallel result: {parallel.invoke(variables)}")


if __name__ == "__main__":
    main()
