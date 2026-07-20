"""One or more tests per example.

Every test runs fully offline (no API key, no network) and asserts that the
example both executes and produces the expected deterministic result. Each test
also calls the example's ``main()`` to prove the script runs end to end.

Run from the repo root:
    pytest -q
"""

from __future__ import annotations

from conftest import load_example


# --- prompt-template -------------------------------------------------------

def test_prompt_template_substitutes_variables():
    mod = load_example("prompt-template")
    tmpl = mod.PromptTemplate("Hello {name}, welcome to {place}.")
    assert tmpl.input_variables == ["name", "place"]
    assert tmpl.format(name="Ada", place="London") == "Hello Ada, welcome to London."


def test_prompt_template_missing_variable_raises():
    mod = load_example("prompt-template")
    import pytest

    with pytest.raises(KeyError):
        mod.PromptTemplate("Hi {name}").format()


def test_prompt_template_main_runs(capsys):
    mod = load_example("prompt-template")
    mod.main()
    assert "prompt-template" in capsys.readouterr().out


# --- chat-models -----------------------------------------------------------

def test_chat_model_is_deterministic():
    mod = load_example("chat-models")
    model = mod.FakeChatModel()
    convo = [mod.system("concise assistant"), mod.human("hi there")]
    first = model.invoke(convo)
    second = model.invoke(convo)
    assert first.role == "ai"
    assert first.content == second.content
    assert "hi there" in first.content
    assert "concise assistant" in first.content


def test_chat_model_main_runs(capsys):
    mod = load_example("chat-models")
    mod.main()
    assert "chat-models" in capsys.readouterr().out


# --- chains ----------------------------------------------------------------

def test_chain_pipes_prompt_model_parser():
    mod = load_example("chains")
    chain = mod.build_chain()
    result = chain.invoke({"topic": "offline testing", "audience": "developers"})
    assert chain.name == "prompt|model|parser"
    assert result == "Write A Tagline About Offline Testing For Developers."


def test_chain_main_runs(capsys):
    mod = load_example("chains")
    mod.main()
    assert "chains" in capsys.readouterr().out


# --- memory ----------------------------------------------------------------

def test_memory_recalls_earlier_turn():
    mod = load_example("memory")
    memory = mod.run_conversation()
    transcript = memory.buffer()
    assert "My name is Ada." in transcript
    assert "Your name is Ada." in transcript


def test_memory_main_runs(capsys):
    mod = load_example("memory")
    mod.main()
    assert "memory" in capsys.readouterr().out


# --- output-parser ---------------------------------------------------------

def test_output_parser_list_and_record():
    mod = load_example("output-parser")
    assert mod.CommaSeparatedListParser().parse("a, b ,c") == ["a", "b", "c"]
    record = mod.KeyValueParser().parse("name: Ada\nborn: 1815\nbad line")
    assert record == {"name": "Ada", "born": 1815}


def test_output_parser_main_runs(capsys):
    mod = load_example("output-parser")
    mod.main()
    assert "output-parser" in capsys.readouterr().out


# --- tools -----------------------------------------------------------------

def test_tools_metadata_and_invocation():
    mod = load_example("tools")
    assert mod.add.name == "add"
    assert mod.add.args == ["a", "b"]
    assert "Add two integers" in mod.add.description
    assert mod.add.invoke(a=2, b=40) == 42
    assert mod.word_count.invoke(text="one two three") == 3


def test_tools_main_runs(capsys):
    mod = load_example("tools")
    mod.main()
    assert "tools" in capsys.readouterr().out


# --- agents ----------------------------------------------------------------

def test_agent_routes_to_correct_tool():
    mod = load_example("agents")
    calc = mod.run_agent("What is 6 * 7?")
    assert calc["chosen_tool"] == "calculator"
    assert calc["observation"] == "42"

    length = mod.run_agent("How long is the phrase 'langchain'?")
    assert length["chosen_tool"] == "length"
    assert length["observation"] == "9"

    rev = mod.run_agent("Please reverse 'offline'.")
    assert rev["chosen_tool"] == "reverse"
    assert rev["observation"] == "enilffo"


def test_agent_main_runs(capsys):
    mod = load_example("agents")
    mod.main()
    assert "agents" in capsys.readouterr().out


# --- document-loader -------------------------------------------------------

def test_document_loader_loads_with_metadata():
    mod = load_example("document-loader")
    docs = mod.load_sample_documents()
    assert len(docs) == 2
    for doc in docs:
        assert doc.page_content
        assert doc.metadata["source"].endswith(".txt")
        assert doc.metadata["chars"] == len(doc.page_content)


def test_document_loader_main_runs(capsys):
    mod = load_example("document-loader")
    mod.main()
    assert "document-loader" in capsys.readouterr().out


# --- text-splitter ---------------------------------------------------------

def test_text_splitter_overlap_and_coverage():
    mod = load_example("text-splitter")
    chunks = mod.split_sample(chunk_size=80, chunk_overlap=20)
    assert len(chunks) == 4
    # First chunk starts at 0, last chunk ends at the full length.
    assert chunks[0].start == 0
    assert chunks[-1].end == len(mod.SAMPLE)
    # Consecutive chunks overlap by the configured amount (step = 60).
    assert chunks[1].start == 60
    assert chunks[0].end == 80


def test_text_splitter_rejects_bad_overlap():
    mod = load_example("text-splitter")
    import pytest

    with pytest.raises(ValueError):
        mod.CharacterTextSplitter(chunk_size=50, chunk_overlap=50)


def test_text_splitter_main_runs(capsys):
    mod = load_example("text-splitter")
    mod.main()
    assert "text-splitter" in capsys.readouterr().out


# --- embeddings ------------------------------------------------------------

def test_embeddings_deterministic_and_normalized():
    mod = load_example("embeddings")
    embedder = mod.HashingEmbedder(dim=64)
    v1 = embedder.embed("cats and dogs")
    v2 = embedder.embed("cats and dogs")
    assert v1 == v2
    assert len(v1) == 64
    # L2-normalized: unit length for non-empty text.
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embeddings_ranks_relevant_document_first():
    mod = load_example("embeddings")
    embedder = mod.HashingEmbedder(dim=128)
    docs = [
        "Cats and dogs are common household pets.",
        "The stock market rallied on strong earnings.",
    ]
    best, score = mod.most_similar("pets like cats and dogs", docs, embedder)
    assert best == docs[0]
    assert score > 0.0


def test_embeddings_main_runs(capsys):
    mod = load_example("embeddings")
    mod.main()
    assert "embeddings" in capsys.readouterr().out
