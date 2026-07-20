"""Memory - LangChain concept example (self-contained, offline).

Chat models are stateless: each call only knows what you pass it. Memory keeps a
running transcript of the conversation and replays it on every turn, so the model
appears to "remember" earlier messages.

This example implements a ``ConversationBufferMemory`` (the same idea as
LangChain's class of that name) plus a tiny deterministic model that answers from
the buffered history. It recalls a fact stated in an earlier turn, all offline
and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationBufferMemory:
    """Stores the full turn-by-turn history of a conversation."""

    turns: list[tuple[str, str]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        """Record a user turn."""
        self.turns.append(("user", text))

    def add_ai(self, text: str) -> None:
        """Record an assistant turn."""
        self.turns.append(("ai", text))

    def buffer(self) -> str:
        """Render the whole history as a single transcript string."""
        return "\n".join(f"{role}: {text}" for role, text in self.turns)


class MemoryAwareModel:
    """A deterministic offline model that reads the conversation buffer.

    It answers "what is my name?" by scanning history for a prior introduction,
    which demonstrates that memory (not the model call itself) carries context
    across turns.
    """

    def respond(self, memory: ConversationBufferMemory, user_text: str) -> str:
        history = memory.buffer().lower()
        if "my name is" in history and "name" in user_text.lower():
            # Recover the name stated earlier in the conversation.
            after = history.split("my name is", 1)[1].strip()
            name = after.split()[0].strip(".,!?").title()
            return f"Your name is {name}."
        return f"Noted: '{user_text}'"


def run_conversation() -> ConversationBufferMemory:
    """Play a short scripted conversation and return the populated memory."""
    memory = ConversationBufferMemory()
    model = MemoryAwareModel()

    for user_text in ["My name is Ada.", "What is my name?"]:
        memory.add_user(user_text)
        reply = model.respond(memory, user_text)
        memory.add_ai(reply)

    return memory


def main() -> None:
    """Entry point: run a conversation and show the model recalling context."""
    memory = run_conversation()
    print("[memory] conversation buffer replayed each turn\n")
    print(memory.buffer())


if __name__ == "__main__":
    main()
