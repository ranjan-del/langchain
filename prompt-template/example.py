"""Prompt template - LangChain concept example (self-contained, offline).

A prompt template is a reusable, parameterized string. You define placeholders
once and fill them with different values to build many concrete prompts without
re-typing boilerplate.

This runs with no API key and no external dependencies. It mirrors the behaviour
of ``langchain_core.prompts.PromptTemplate`` (``{variable}`` substitution and
``.format(**kwargs)``) using only the standard library, so the concept is clear
and the output is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A reusable prompt with ``{name}`` placeholders.

    Attributes:
        template: the template string, e.g. ``"Translate {text} to {language}."``
    """

    template: str

    @property
    def input_variables(self) -> list[str]:
        """Return the placeholder names declared in the template, in order."""
        seen: list[str] = []
        for name in re.findall(r"\{(\w+)\}", self.template):
            if name not in seen:
                seen.append(name)
        return seen

    def format(self, **kwargs: str) -> str:
        """Fill every placeholder. Raises if a required variable is missing."""
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise KeyError(f"missing prompt variables: {missing}")
        return self.template.format(**kwargs)

    def partial(self, **kwargs: str) -> "PromptTemplate":
        """Pre-fill some variables, returning a new template for the rest."""
        filled = self.template
        for key, value in kwargs.items():
            filled = filled.replace("{" + key + "}", value)
        return PromptTemplate(filled)


def build_examples() -> list[tuple[str, str]]:
    """Return a list of (description, rendered_prompt) demonstrations."""
    translate = PromptTemplate("Translate '{text}' into {language}.")
    summarize = PromptTemplate(
        "You are a {role}. Summarize the following in {n} words:\n{content}"
    )

    results = [
        (
            "simple substitution",
            translate.format(text="good morning", language="French"),
        ),
        (
            "multi-variable prompt",
            summarize.format(
                role="teacher",
                n="5",
                content="LangChain composes prompts, models and parsers.",
            ),
        ),
    ]

    # Partial application: bind the role now, supply the rest later.
    reviewer = summarize.partial(role="code reviewer")
    results.append(
        (
            "partial then format",
            reviewer.format(n="3", content="The function lacks a docstring."),
        )
    )
    return results


def main() -> None:
    """Entry point: render a few prompts from templates and print them."""
    print("[prompt-template] rendering reusable prompts\n")
    for description, rendered in build_examples():
        print(f"- {description}:")
        print(f"    {rendered}\n")


if __name__ == "__main__":
    main()
