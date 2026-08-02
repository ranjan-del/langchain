"""Prompt template - LangChain concept example (self-contained, offline).

A prompt template is a reusable, parameterized string. You define placeholders
once and fill them with different values to build many concrete prompts without
re-typing boilerplate.

This runs with no API key and no external dependencies. It mirrors the behaviour
of ``langchain_core.prompts.PromptTemplate`` (``{variable}`` substitution and
``.format(**kwargs)``) and ``ChatPromptTemplate`` (the same substitution applied
across a list of role-tagged messages) using only the standard library, so the
concept is clear and the output is deterministic.
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


@dataclass
class ChatPromptTemplate:
    """A list of role-tagged message templates rendered as one conversation.

    Chat models do not take a single string, they take a list of ``(role, text)``
    messages. ``ChatPromptTemplate`` therefore wraps several
    :class:`PromptTemplate` instances (one per message) and fills them all from
    the same variable dict. This is why LangChain ships a separate chat prompt
    class rather than reusing the plain string template.

    Attributes:
        messages: ordered ``(role, PromptTemplate)`` pairs.
    """

    messages: list[tuple[str, PromptTemplate]]

    @classmethod
    def from_messages(cls, messages: list[tuple[str, str]]) -> "ChatPromptTemplate":
        """Build from raw ``(role, template_string)`` pairs, as LangChain does."""
        return cls([(role, PromptTemplate(text)) for role, text in messages])

    @property
    def input_variables(self) -> list[str]:
        """Union of every message's placeholders, in first-seen order."""
        seen: list[str] = []
        for _role, template in self.messages:
            for name in template.input_variables:
                if name not in seen:
                    seen.append(name)
        return seen

    def format_messages(self, **kwargs: str) -> list[tuple[str, str]]:
        """Render every message, returning ``(role, text)`` pairs.

        Each message is formatted with the *same* kwargs, so a variable used in
        both the system and the human message only has to be supplied once.
        """
        return [(role, template.format(**kwargs)) for role, template in self.messages]


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


def build_chat_prompt() -> ChatPromptTemplate:
    """Return a two-message chat prompt sharing one variable across messages."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {role} who answers in {style} sentences."),
            ("human", "Explain {topic} to me."),
        ]
    )


def main() -> None:
    """Entry point: render a few prompts from templates and print them."""
    print("[prompt-template] rendering reusable prompts\n")
    for description, rendered in build_examples():
        print(f"- {description}:")
        print(f"    {rendered}\n")

    chat_prompt = build_chat_prompt()
    print(f"- chat prompt (variables: {chat_prompt.input_variables}):")
    for role, text in chat_prompt.format_messages(
        role="tutor", style="short", topic="prompt templates"
    ):
        print(f"    {role:>6}: {text}")


if __name__ == "__main__":
    main()
