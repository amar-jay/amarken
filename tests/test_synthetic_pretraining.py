from src.distillation.synthetic_pretraining import category_for, language_for, make_spec, valid


def test_mixture_is_exact_over_repeating_schedule():
    assert [language_for(i) for i in range(20)].count("en") == 9
    assert [language_for(i) for i in range(20)].count("tr") == 9
    assert [language_for(i) for i in range(20)].count("bilingual") == 2
    counts={name:sum(category_for(i)==name for i in range(100)) for name in {category_for(i) for i in range(100)}}
    assert counts=={"conversation":30,"contextual_qa":20,"transformation":15,"reasoning":15,"general_qa":10,"multi_turn":10}


def test_specs_are_deterministic_unique_and_no_code():
    first=[make_spec(i,91423) for i in range(500)]
    assert first==[make_spec(i,91423) for i in range(500)]
    assert len({row["id"] for row in first})==500
    assert all("code" not in row["category"] for row in first)


def test_validation_enforces_grounding_language_and_code_filter():
    config={"min_words":2,"max_words":72}
    spec={"language":"tr","required":["20"]}
    assert valid("Toplam 20 eşya vardır.",spec,config)[0]
    assert not valid("There are 20 items.",spec,config)[0]
    assert not valid("```python\nprint(20)\n```",spec,config)[0]
