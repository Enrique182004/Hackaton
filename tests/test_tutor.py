from __future__ import annotations

import threading

import pytest

from biobench_sar.rag.quiz_bank import QuizProblem
from biobench_sar.tutor import (
    DEFAULT_QUIZ_LENGTH,
    MAX_TURNS,
    MODES,
    TutorSession,
    detect_learn_request,
    detect_mode,
    detect_offer_reply,
    is_quiz_stop,
)


def _problems(*texts: str) -> list[QuizProblem]:
    return [QuizProblem(t, f"answer {i}", "equations", 1, "w.md") for i, t in enumerate(texts, 1)]


def test_modes() -> None:
    assert MODES == ("help", "practice", "quiz")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("quiz me", "quiz"),
        ("Quiz me on fractions!", "quiz"),
        ("Can we start a quiz?", "quiz"),
        ("let's do a quiz", "quiz"),
        ("stop the quiz", "help"),
        ("Please end the quiz.", "help"),
        ("no more quizzes", "help"),
        ("give me a similar problem", "practice"),
        ("another one like this", "practice"),
        ("Another problem like that please", "practice"),
        ("I need more practice", "practice"),
        ("help", "help"),
        ("Can you help me?", "help"),
        ("what's the next step?", "help"),
        ("I'm stuck", "help"),
        ("I’m stuck", "help"),
        ("give me a hint", "help"),
        ("what is three times four", None),
        ("this is helpful", None),
        ("", None),
    ],
)
def test_detect_mode(text: str, expected: str | None) -> None:
    assert detect_mode(text) == expected


def test_detect_mode_prefers_practice_over_a_plain_help_word() -> None:
    assert detect_mode("help me with another one like this") == "practice"


@pytest.mark.parametrize(("text", "expected"), [("stop the quiz", True), ("I'm stuck", False), ("help", False)])
def test_is_quiz_stop(text: str, expected: bool) -> None:
    assert is_quiz_stop(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to learn how to solve fractions", "solving fractions"),
        ("I want to learn how to solve a fraction", "solving a fraction"),
        ("I'd like to learn about fractions", "fractions"),
        ("teach me equations", "equations"),
        ("Can you teach me how to divide fractions please?", "dividing fractions"),
        ("teach me to simplify", "simplifying"),
        ("How do I add fractions?", "adding fractions"),
        ("how do you multiply decimals", "multiplying decimals"),
        ("show me how to graph a line", "graphing a line"),
        ("explain how to find the area of a triangle", "finding the area of a triangle"),
        ("explain fractions to me", "fractions"),
        ("teach me about the order of operations", "the order of operations"),
    ],
)
def test_detect_learn_request(text: str, expected: str) -> None:
    assert detect_learn_request(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "how do I solve this?",  # about the board: help mode answers it
        "how do I do the next step",
        "how do I find x",
        "teach me how to do that",
        "can you explain this step",
        "I want to learn",
        "what did we learn today?",
        "what is three plus four",
        "is this right?",
        "",
    ],
)
def test_detect_learn_request_ignores_board_questions(text: str) -> None:
    assert detect_learn_request(text) is None


def test_learn_topic_is_short() -> None:
    topic = detect_learn_request("teach me " + " ".join(["fractions"] * 20))
    assert topic is not None and len(topic.split()) == 8


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("make one up", "generate"),
        ("Make one up for me!", "generate"),
        ("give me one", "generate"),
        ("you choose", "generate"),
        ("yes make one", "generate"),
        ("surprise me", "generate"),
        ("I don't have one", "generate"),
        ("no", "generate"),
        ("no I don't", "generate"),
        ("I have one", "own"),
        ("I've got my own", "own"),
        ("I'll write mine", "own"),
        ("yes I do", "own"),
        ("I have an exercise from homework", "own"),
        ("let me write it on the board", "own"),
        ("what is a fraction", None),
        ("hmm", None),
        ("", None),
    ],
)
def test_detect_offer_reply(text: str, expected: str | None) -> None:
    assert detect_offer_reply(text) == expected


# ---- session: mode ------------------------------------------------------------


def test_new_session() -> None:
    t = TutorSession()
    assert t.mode == "help" and t.learn_topic is None and not t.offer_pending and t.turns() == []
    assert t.current() is None and not t.dictated and not t.quiz_active
    assert t.to_dict() == {"mode": "help", "learn_topic": None, "quiz": None}


def test_set_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        TutorSession().set_mode("chat")


def test_set_mode_quiz_does_not_start_a_quiz() -> None:
    t = TutorSession()
    t.set_mode("quiz")
    assert t.mode == "quiz" and not t.quiz_active and t.to_dict()["quiz"] is None


def test_leaving_quiz_mode_stops_the_quiz() -> None:
    t = TutorSession()
    t.start_quiz(_problems("1 + 1"))
    t.set_mode("practice")
    assert t.mode == "practice" and not t.quiz_active and t.to_dict()["quiz"] is None


# ---- session: quiz ------------------------------------------------------------


def test_quiz_runs_through_its_problems_and_finishes() -> None:
    t = TutorSession()
    t.set_mode("practice")
    t.start_quiz(_problems("1 + 1", "2x = 8", "3x + 5 = 20"))
    assert t.mode == "quiz"
    assert t.to_dict()["quiz"] == {"active": True, "number": 1, "total": 3, "score": 0, "problem": "1 + 1"}
    assert t.next_problem() is not None and t.next_problem().problem == "2x = 8"  # type: ignore[union-attr]
    t.mark_dictated()
    assert t.dictated
    t.record(True)
    assert not t.dictated, "a new problem has to be read out"
    assert t.current().problem == "2x = 8"  # type: ignore[union-attr]
    t.record(False)
    assert t.current().problem == "2x = 8", "a wrong answer stays on the problem"  # type: ignore[union-attr]
    t.record(True)
    assert t.to_dict()["quiz"] == {"active": True, "number": 3, "total": 3, "score": 1, "problem": "3x + 5 = 20"}
    assert t.next_problem() is None
    t.record(True)
    assert not t.quiz_active and t.mode == "help"
    assert t.to_dict()["quiz"] == {"active": False, "number": 3, "total": 3, "score": 2, "problem": None}
    t.record(True)
    assert t.score == 2, "a finished quiz ignores further answers"


def test_quiz_without_problems_lets_the_model_invent_them() -> None:
    t = TutorSession()
    t.start_quiz([])
    assert t.quiz_total == DEFAULT_QUIZ_LENGTH and t.current() is None and t.next_problem() is None
    for _ in range(DEFAULT_QUIZ_LENGTH):
        assert t.quiz_active
        t.record(True)
    assert not t.quiz_active and t.score == DEFAULT_QUIZ_LENGTH


def test_start_quiz_restarts() -> None:
    t = TutorSession()
    t.start_quiz(_problems("1 + 1", "2 + 2"))
    t.mark_dictated()
    t.record(True)
    t.start_quiz(_problems("5 - 3"))
    assert t.to_dict()["quiz"] == {"active": True, "number": 1, "total": 1, "score": 0, "problem": "5 - 3"}
    assert not t.dictated


def test_stop_quiz() -> None:
    t = TutorSession()
    t.start_quiz(_problems("1 + 1"))
    t.mark_dictated()
    t.stop_quiz()
    assert t.mode == "help" and not t.quiz_active and not t.dictated and t.to_dict()["quiz"] is None
    t.mark_dictated()
    assert not t.dictated, "nothing to dictate without a quiz"


def test_record_and_dictate_without_a_quiz_are_ignored() -> None:
    t = TutorSession()
    t.record(True)
    t.mark_dictated()
    assert t.to_dict() == {"mode": "help", "learn_topic": None, "quiz": None}


def test_start_quiz_copies_the_problem_list() -> None:
    problems = _problems("1 + 1")
    t = TutorSession()
    t.start_quiz(problems)
    problems.clear()
    assert t.current() is not None


# ---- session: learn and memory ---------------------------------------------


def test_learning_offer() -> None:
    t = TutorSession()
    t.start_learning("fractions")
    assert t.learn_topic == "fractions" and t.offer_pending
    assert t.to_dict()["learn_topic"] == "fractions"
    t.resolve_offer()
    assert t.learn_topic == "fractions" and not t.offer_pending


def test_turns_keep_the_last_three_oldest_first() -> None:
    t = TutorSession()
    for n in range(5):
        t.add_turn(f"q{n}", f"a{n}")
    assert t.turns() == [{"q": f"q{n}", "a": f"a{n}"} for n in range(5 - MAX_TURNS, 5)]
    t.turns()[0]["q"] = "changed"
    assert t.turns()[0]["q"] == "q2", "a copy"


def test_turns_are_truncated() -> None:
    t = TutorSession()
    t.add_turn("q" * 1000, "a" * 1000)
    [turn] = t.turns()
    assert len(turn["q"]) < 1000 and len(turn["a"]) < 1000


def test_session_is_thread_safe() -> None:
    t = TutorSession()
    t.start_quiz([])
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for n in range(200):
                t.add_turn(str(n), str(n))
                t.to_dict()
                t.mark_dictated()
        except Exception as exc:  # pragma: no cover - only on a race
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    [th.start() for th in threads]
    for _ in range(DEFAULT_QUIZ_LENGTH):
        t.record(True)
    [th.join() for th in threads]
    assert not errors and not t.quiz_active and len(t.turns()) == MAX_TURNS
