"""Query understanding — heuristic intent classification. (Requires runtime deps.)"""

from app.services.query_understanding import _heuristic_intent, _normalize


def test_intents():
    assert _heuristic_intent("Summarize the maternity policy") == "summarize"
    assert _heuristic_intent("Compare maternity vs paternity leave") == "compare"
    assert _heuristic_intent("hello there") == "chitchat"
    assert _heuristic_intent("What is the minimum GPA for CS?") == "factual"


def test_normalize_collapses_whitespace():
    assert _normalize("  what   is\tthe   GPA?  ") == "what is the GPA?"
