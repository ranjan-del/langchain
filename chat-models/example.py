"""Chat model - LangChain concept example (self-contained, offline).

A chat model takes a list of role-tagged messages (system, human, ai) and
returns an assistant message. Real deployments call a hosted model behind an
API key and a network call.

So the whole repo stays runnable with no key, this example uses a deterministic
"echo" chat model as the offline fallback. Its replies are reproducible and
depend only on the input, which is exactly what tests need. The public shape
(``ChatModel.invoke(messages) -> AIMessage``) mirrors LangChain's chat model
interface. In a real setup you would set a provider API key to swap in a hosted
model behind the same call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Message:
    """A single chat message with a role and text content."""

    role: str  # "system", "human" or "ai"
    content: str


def system(text: str) -> Message:
    """Build a system message (sets the assistant's behaviour)."""
    return Message("system", text)


def human(text: str) -> Message:
    """Build a human (user) message."""
    return Message("human", text)


class FakeChatModel:
    """Deterministic offline chat model.

    It reads the system instruction and the latest human message, then returns
    a reproducible reply. This is a stand-in for a hosted model so the example
    runs with no API key and no network.
    """

    def __init__(self, name: str = "fake-echo-chat") -> None:
        self.name = name

    def invoke(self, messages: list[Message]) -> Message:
        """Return an AI message computed deterministically from the input."""
        persona = next((m.content for m in messages if m.role == "system"), None)
        last_human = next(
            (m.content for m in reversed(messages) if m.role == "human"), ""
        )
        prefix = f"[{persona}] " if persona else ""
        reply = f"{prefix}You said: '{last_human}'. Here is a helpful reply."
        return Message("ai", reply)


def get_chat_model() -> FakeChatModel:
    """Return a chat model.

    In a real project this would return a hosted model when an API key is
    present. Offline (the default here) it returns the deterministic fake.
    """
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        # A real integration would construct the hosted client here.
        return FakeChatModel(name="fake-echo-chat (key present, still offline demo)")
    return FakeChatModel()


def main() -> None:
    """Entry point: invoke the chat model with a small conversation."""
    model = get_chat_model()
    conversation = [
        system("concise assistant"),
        human("What is a chat model?"),
    ]
    print(f"[chat-models] model = {model.name}\n")
    for msg in conversation:
        print(f"  {msg.role:>6}: {msg.content}")
    response = model.invoke(conversation)
    print(f"  {response.role:>6}: {response.content}")


if __name__ == "__main__":
    main()
