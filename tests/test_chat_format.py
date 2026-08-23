from src.data.chat_format import render_chat
from src.data.dataset import PackedConversationDataset
from src.tokenization.tokenizer import load_tokenizer


def test_canonical_chat_template_matches_tokenizer_corpus_format():
    rendered = render_chat([
        {"role": "user", "content": " Hello "},
        {"role": "assistant", "content": " Merhaba "},
    ])

    assert rendered is not None
    assert rendered.text == (
        "<|user|>: Hello \n<|end|>\n<|assistant|>: Merhaba \n<|end|>"
    )
    assert rendered.assistant_spans[0].start == rendered.text.index(": Merhaba") + 1
    assert rendered.text[rendered.assistant_spans[0].start:] == " Merhaba \n<|end|>"


def test_runtime_encoding_is_exactly_the_canonical_whole_text_tokenization():
    tokenizer = load_tokenizer("artifacts/tokenizers/v3/tiktoken-tr-bpe-12k.json")
    record = {"id": "one", "messages": [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Merhaba dünya"},
    ]}
    rendered = render_chat(record["messages"])
    assert rendered is not None
    dataset = object.__new__(PackedConversationDataset)
    dataset.tokenizer = tokenizer

    ids, labels = dataset._encode(record)
    assert ids == tokenizer.encode(rendered.text)
    assert labels.count(-100) < len(labels)
