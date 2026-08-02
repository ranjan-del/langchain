"""Memory - LangChain concept example (self-contained, offline).

Chat models are stateless: each call only knows what you pass it. Memory keeps a
running transcript of the conversation and replays it on every turn, so the model
appears to "remember" earlier messages.

This example implements three memory strategies that mirror LangChain's classes
of the same names, plus a tiny deterministic model that answers from history:

* ``ConversationBufferMemory``       - keep everything.
* ``ConversationBufferWindowMemory`` - keep only the last k turns.
* ``ConversationSummaryMemory``      - keep a compressed summary plus recent turns.

They exist because a full transcript eventually overflows the context window and
costs money per token, so real applications trade recall against size. All three
run offline and reproducibly.
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


@dataclass
class ConversationBufferWindowMemory(ConversationBufferMemory):
    """Keep only the most recent ``k`` turns.

    A sliding window bounds the prompt size no matter how long the chat runs.
    The cost is honest and worth stating: anything older than the window is
    genuinely forgotten, so facts stated early cannot be recalled later.
    """

    k: int = 4

    def buffer(self) -> str:
        """Render only the last ``k`` turns as a transcript."""
        recent = self.turns[-self.k :] if self.k > 0 else []
        return "\n".join(f"{role}: {text}" for role, text in recent)


@dataclass
class ConversationSummaryMemory(ConversationBufferMemory):
    """Compress older turns into a one-line summary, keep the last ``k`` verbatim.

    This is the middle ground between the two strategies above: the prompt stays
    bounded, but a lossy trace of the early conversation survives. LangChain uses
    an LLM to write that summary; to stay offline and deterministic this version
    summarises by counting the dropped turns and listing who spoke.
    """

    k: int = 2

    def summary(self) -> str:
        """Return the one-line summary of the turns that fell out of the window."""
        older = self.turns[: -self.k] if self.k > 0 else list(self.turns)
        if not older:
            return ""
        speakers = sorted({role for role, _ in older})
        return f"[summary of {len(older)} earlier turn(s) by {', '.join(speakers)}]"

    def buffer(self) -> str:
        """Render the summary line (if any) followed by the recent turns."""
        recent = self.turns[-self.k :] if self.k > 0 else []
        lines = [f"{role}: {text}" for role, text in recent]
        summary = self.summary()
        return "\n".join(([summary] if summary else []) + lines)


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


LONG_SCRIPT = [
    "My name is Ada.",
    "I work on analytical engines.",
    "I also write about mathematics.",
    "What is my name?",
]


def replay(memory: ConversationBufferMemory) -> ConversationBufferMemory:
    """Run ``LONG_SCRIPT`` against ``memory`` and return it populated.

    The same script is fed to each memory strategy so the only variable is what
    the memory chooses to retain. That is what makes the recall difference below
    a fair comparison rather than a coincidence.
    """
    model = MemoryAwareModel()
    for user_text in LONG_SCRIPT:
        memory.add_user(user_text)
        memory.add_ai(model.respond(memory, user_text))
    return memory


def compare_memories() -> dict[str, str]:
    """Return each strategy's final reply to the last question in the script."""
    strategies: dict[str, ConversationBufferMemory] = {
        "buffer": ConversationBufferMemory(),
        "window(k=2)": ConversationBufferWindowMemory(k=2),
        "summary(k=2)": ConversationSummaryMemory(k=2),
    }
    # The last recorded turn is the AI's answer to "What is my name?".
    return {name: replay(memory).turns[-1][1] for name, memory in strategies.items()}


def main() -> None:
    """Entry point: run a conversation and show the model recalling context."""
    memory = run_conversation()
    print("[memory] conversation buffer replayed each turn\n")
    print(memory.buffer())

    print("\n  same 4-turn script, three memory strategies:")
    for name, answer in compare_memories().items():
        print(f"    {name:<13} -> {answer}")
    print("\n  only the full buffer still holds the introduction; the window")
    print("  dropped it and the lossy summary did not preserve the name")


if __name__ == "__main__":
    main()
