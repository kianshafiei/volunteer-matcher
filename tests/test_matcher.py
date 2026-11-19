import pytest
from fastapi.testclient import TestClient
from vmatch.api import app
from vmatch.matcher import Matcher, tokenize
from vmatch.data import load


@pytest.fixture(scope="module")
def m():
    return Matcher(load())


def test_data_clean():
    d = load()
    assert d.doc_id.is_unique
    assert not d.text.str.contains("http").any()


def test_tokenize_drops_stopwords():
    assert tokenize("I want to help the kids") == ["want", "help", "kids"]


def test_semantic_match_without_shared_words(m):
    # "pets" and "animal foster" share no words, so only the embedding can find it
    top = [r.title for r in m.search("I'd like to look after pets", 5, "dense")]
    assert any("Animal" in t for t in top)


def test_hard_constraint_filters(m):
    res = m.search("I can only do a single day, one time", 10, "full")
    assert all(r.recurrence == "onetime" for r in res)


def test_diversity_limits_one_org(m):
    res = m.search("recycling and swap events", 10, "full")
    orgs = [r.org for r in res]
    assert max(orgs.count(o) for o in set(orgs)) <= 3


def test_every_result_has_reason(m):
    assert all(r.why for r in m.search("tutor maths after school", 5))


def test_empty_query_rejected(m):
    with pytest.raises(ValueError):
        m.search("   ")


def test_api():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/match", json={"text": "help at a food bank", "k": 3})
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    assert c.post("/match", json={"text": "food", "method": "magic"}).status_code == 422
