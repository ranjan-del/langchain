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


def test_prompt_template_partial_binds_some_variables():
    mod = load_example("prompt-template")
    tmpl = mod.PromptTemplate("A {a} and a {b}.")
    bound = tmpl.partial(a="cat")
    assert bound.input_variables == ["b"]
    assert bound.format(b="dog") == "A cat and a dog."


def test_chat_prompt_template_shares_variables_across_messages():
    mod = load_example("prompt-template")
    prompt = mod.ChatPromptTemplate.from_messages(
        [("system", "You are {role}."), ("human", "As {role}, explain {topic}.")]
    )
    # "role" appears in both messages but is only declared once.
    assert prompt.input_variables == ["role", "topic"]
    messages = prompt.format_messages(role="a tutor", topic="vectors")
    assert messages == [
        ("system", "You are a tutor."),
        ("human", "As a tutor, explain vectors."),
    ]


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


def test_chat_model_stream_reassembles_to_invoke():
    mod = load_example("chat-models")
    model = mod.FakeChatModel()
    convo = [mod.human("stream this please")]
    tokens = list(model.stream(convo))
    assert len(tokens) > 1
    assert "".join(tokens) == model.invoke(convo).content


def test_chat_model_batch_keeps_conversations_isolated():
    mod = load_example("chat-models")
    model = mod.FakeChatModel()
    replies = model.batch(
        [[mod.human("first")], [mod.system("pirate"), mod.human("second")]]
    )
    assert len(replies) == 2
    assert "first" in replies[0].content and "pirate" not in replies[0].content
    assert "second" in replies[1].content and "pirate" in replies[1].content


def test_chat_model_ai_helper_builds_assistant_turn():
    mod = load_example("chat-models")
    assert mod.ai("prior answer") == mod.Message("ai", "prior answer")


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


def test_chain_batch_matches_individual_invokes():
    mod = load_example("chains")
    chain = mod.build_chain()
    inputs = [
        {"topic": "vector search", "audience": "students"},
        {"topic": "chunking", "audience": "librarians"},
    ]
    assert chain.batch(inputs) == [chain.invoke(i) for i in inputs]


def test_chain_parallel_fans_out_to_named_branches():
    mod = load_example("chains")
    chain = mod.build_parallel_chain()
    result = chain.invoke({"topic": "offline testing", "audience": "developers"})
    assert set(result) == {"parsed", "raw_length"}
    assert result["parsed"] == "Write A Tagline About Offline Testing For Developers."
    # raw_length is measured on the un-parsed model output, so it differs.
    assert result["raw_length"] > len(result["parsed"])


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


def test_memory_window_forgets_older_turns():
    mod = load_example("memory")
    memory = mod.ConversationBufferWindowMemory(k=2)
    for text in ["one", "two", "three", "four"]:
        memory.add_user(text)
    buffer = memory.buffer()
    assert "one" not in buffer and "two" not in buffer
    assert "three" in buffer and "four" in buffer


def test_memory_summary_keeps_a_trace_of_dropped_turns():
    mod = load_example("memory")
    memory = mod.ConversationSummaryMemory(k=2)
    for text in ["one", "two", "three", "four"]:
        memory.add_user(text)
    buffer = memory.buffer()
    assert buffer.startswith("[summary of 2 earlier turn(s) by user]")
    assert "three" in buffer and "four" in buffer
    assert "one" not in buffer


def test_memory_summary_has_no_summary_line_when_nothing_dropped():
    mod = load_example("memory")
    memory = mod.ConversationSummaryMemory(k=2)
    memory.add_user("only turn")
    assert memory.summary() == ""
    assert memory.buffer() == "user: only turn"


def test_memory_only_full_buffer_recalls_the_name():
    mod = load_example("memory")
    answers = mod.compare_memories()
    assert answers["buffer"] == "Your name is Ada."
    # The window dropped the introduction and the lossy summary did not keep it.
    assert answers["window(k=2)"] != "Your name is Ada."
    assert answers["summary(k=2)"] != "Your name is Ada."


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


def test_output_parser_json_strips_markdown_fence():
    mod = load_example("output-parser")
    fenced = '```json\n{"name": "Ada", "born": 1815}\n```'
    assert mod.JsonOutputParser().parse(fenced) == {"name": "Ada", "born": 1815}
    # Bare JSON, with no fence, must work too.
    assert mod.JsonOutputParser().parse('{"a": 1}') == {"a": 1}


def test_output_parser_json_raises_on_bad_output():
    mod = load_example("output-parser")
    import pytest

    with pytest.raises(mod.OutputParserException):
        mod.JsonOutputParser().parse("sorry, I cannot answer")
    # Valid JSON but the wrong shape is still a parse failure.
    with pytest.raises(mod.OutputParserException):
        mod.JsonOutputParser().parse("[1, 2, 3]")


def test_output_parsers_expose_format_instructions():
    mod = load_example("output-parser")
    parsers = [
        mod.CommaSeparatedListParser(),
        mod.KeyValueParser(),
        mod.JsonOutputParser(),
    ]
    for parser in parsers:
        instructions = parser.get_format_instructions()
        assert isinstance(instructions, str) and instructions.strip()


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


def test_tools_args_schema_derives_json_types_from_hints():
    mod = load_example("tools")
    assert mod.add.args_schema == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    # A parameter with a default is optional, so it is not in "required".
    repeat_schema = mod.repeat.args_schema
    assert repeat_schema["properties"]["times"] == {"type": "integer"}
    assert repeat_schema["required"] == ["text"]


def test_tools_reject_bad_arguments():
    mod = load_example("tools")
    import pytest

    with pytest.raises(mod.ToolInvocationError):
        mod.add.invoke(a=1)
    with pytest.raises(mod.ToolInvocationError):
        mod.add.invoke(a=1, b=2, c=3)


def test_tools_default_argument_is_applied():
    mod = load_example("tools")
    assert mod.repeat.invoke(text="hi") == "hi hi"
    assert mod.repeat.invoke(text="hi", times=3) == "hi hi hi"


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


def test_agent_admits_when_no_tool_matches():
    mod = load_example("agents")
    assert mod.route("What is the capital of France?") is None
    trace = mod.run_agent("What is the capital of France?")
    assert trace["chosen_tool"] == "none"
    assert trace["answer"] == "No tool matched this request."


def test_agent_calculator_handles_all_four_operators():
    mod = load_example("agents")
    assert mod.run_agent("What is 100 - 58?")["observation"] == "42"
    assert mod.run_agent("What is 84 / 2?")["observation"] == "42"
    assert mod.run_agent("What is 40 + 2?")["observation"] == "42"
    assert mod.run_agent("What is 6 * 7?")["observation"] == "42"


def test_agent_calculator_survives_division_by_zero():
    mod = load_example("agents")
    assert mod.run_agent("What is 1 / 0?")["observation"] == "division by zero"


def test_agent_explicit_intent_beats_arithmetic_pattern():
    mod = load_example("agents")
    # The query contains "3+4" but the user asked to reverse it.
    assert mod.route("reverse '3+4'") == "reverse"


def test_agent_loop_feeds_observation_into_next_step():
    mod = load_example("agents")
    result = mod.run_agent_loop("Reverse 'langchain' and tell me how long it is.")
    steps = result["steps"]
    assert [step.tool for step in steps] == ["reverse", "length"]
    assert steps[0].observation == "niahcgnal"
    # Step two measured the reversed string, not the original query.
    assert steps[1].observation == "9"
    assert result["answer"] == "The answer is 9."


def test_agent_loop_returns_nothing_when_no_tool_applies():
    mod = load_example("agents")
    result = mod.run_agent_loop("What is the capital of France?")
    assert result["steps"] == []
    assert result["answer"] == "No tool matched this request."


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


def test_document_loader_csv_yields_one_document_per_row():
    mod = load_example("document-loader")
    docs = mod.load_sample_csv()
    assert len(docs) == 2
    assert [doc.metadata["row"] for doc in docs] == [1, 2]
    assert docs[0].page_content == "name: Ada Lovelace\nrole: mathematician"
    assert docs[1].page_content == "name: Alan Turing\nrole: logician"


def test_document_loader_csv_cleans_up_its_temp_dir():
    mod = load_example("document-loader")
    import os

    docs = mod.load_sample_csv()
    # The documents survive in memory, but the file on disk is gone.
    assert docs and not os.path.exists(str(docs[0].metadata["source"]))


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


def test_recursive_splitter_respects_chunk_size():
    mod = load_example("text-splitter")
    chunks = mod.split_sample_recursively(chunk_size=80, chunk_overlap=20)
    assert chunks
    # chunk_size is a hard bound, even once overlap has been carried over.
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_recursive_splitter_prefers_sentence_boundaries():
    mod = load_example("text-splitter")
    chunks = mod.split_sample_recursively(chunk_size=80, chunk_overlap=0)
    # With no overlap, every chunk should end at a sentence boundary rather
    # than mid-word, which is exactly what the blind character splitter cannot do.
    assert all(chunk.rstrip().endswith(".") for chunk in chunks)


def test_recursive_splitter_covers_all_the_text():
    mod = load_example("text-splitter")
    chunks = mod.split_sample_recursively(chunk_size=80, chunk_overlap=0)
    # Zero overlap means the chunks should rejoin into the original exactly.
    assert "".join(chunks) == mod.SAMPLE


def test_recursive_splitter_falls_back_to_hard_cut():
    mod = load_example("text-splitter")
    splitter = mod.RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    # No separator occurs in this text, so the last-resort cut must still apply.
    chunks = splitter.split_text("a" * 25)
    assert chunks == ["aaaaaaaaaa", "aaaaaaaaaa", "aaaaa"]


def test_recursive_splitter_handles_empty_text():
    mod = load_example("text-splitter")
    assert mod.RecursiveCharacterTextSplitter().split_text("") == []


def test_recursive_splitter_rejects_bad_overlap():
    mod = load_example("text-splitter")
    import pytest

    with pytest.raises(ValueError):
        mod.RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)


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


def test_vector_store_returns_top_k_in_score_order():
    mod = load_example("embeddings")
    store = mod.InMemoryVectorStore(mod.HashingEmbedder(dim=128))
    store.add_texts(
        [
            "Cats and dogs are common household pets.",
            "The stock market rallied on strong earnings.",
            "Kittens and puppies are baby pets that people love.",
        ]
    )
    hits = store.similarity_search("pets like cats and dogs", k=2)
    assert len(hits) == 2
    assert hits[0][0] == "Cats and dogs are common household pets."
    # Results are ordered best first.
    assert hits[0][1] >= hits[1][1]


def test_vector_store_embeds_once_at_insert_time():
    mod = load_example("embeddings")
    store = mod.InMemoryVectorStore(mod.HashingEmbedder(dim=32))
    store.add_texts(["alpha", "beta"])
    assert len(store.vectors) == 2
    # The stored vector is the embedding of the stored text.
    assert store.vectors[0] == store.embedder.embed("alpha")


def test_embeddings_empty_text_returns_zero_vector():
    mod = load_example("embeddings")
    vector = mod.HashingEmbedder(dim=16).embed("")
    # No tokens means no norm to divide by; it must not raise.
    assert vector == [0.0] * 16


def test_embeddings_main_runs(capsys):
    mod = load_example("embeddings")
    mod.main()
    assert "embeddings" in capsys.readouterr().out
