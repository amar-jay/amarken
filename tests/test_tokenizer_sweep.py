import json

import sentencepiece as spm

from src.tokenization.sweep import _language_metrics, _morphology_metrics, _stress_metrics


def test_metrics_count_fertility_byte_fallback_and_roundtrip(tmp_path):
    train = tmp_path / "train.txt"
    train.write_text(("hello world\nmerhaba dünya\nevlerimizden geliyoruz\n" * 200), encoding="utf-8")
    prefix = tmp_path / "tiny"
    spm.SentencePieceTrainer.train(
        input=str(train),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=320,
        byte_fallback=True,
        character_coverage=1.0,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    metrics = _language_metrics(processor, (train,))
    stress_rate, failures = _stress_metrics(processor)
    assert metrics.fertility > 0
    assert metrics.unknown_tokens == metrics.roundtrip_failures == 0
    assert stress_rate > 0 and failures == 0


def test_morphology_metric_reports_each_probe(tmp_path):
    train = tmp_path / "train.txt"
    train.write_text(("ev evler evlerimizden kitap kitaplarımızdan\n" * 200), encoding="utf-8")
    prefix = tmp_path / "tiny"
    spm.SentencePieceTrainer.train(
        input=str(train), model_prefix=str(prefix), model_type="bpe", vocab_size=300,
        byte_fallback=True, character_coverage=1.0, hard_vocab_limit=False,
        minloglevel=2,
    )
    morphology = tmp_path / "morph.json"
    morphology.write_text(json.dumps({"families": [{"lemma": "ev", "forms": ["ev", "evlerimizden"]}]}), encoding="utf-8")
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    metrics = _morphology_metrics(processor, morphology)
    assert metrics.words == 2 and len(metrics.family_details) == 1
    assert metrics.tokens >= metrics.words
