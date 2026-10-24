from __future__ import annotations

import random
import threading

import pytest

from biobench_sar.rag.quiz_bank import INVENTED_SOURCE, QuizProblem, solve_problem
from biobench_sar.tutor import (
    DEFAULT_QUIZ_LENGTH,
    MAX_PROBLEM_CHARS,
    MAX_TURNS,
    MODES,
    TutorSession,
    detect_learn_request,
    detect_mode,
    detect_offer_reply,
    is_quiz_stop,
    is_repeat_request,
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
        ("I want a quiz", "quiz"),
        ("can I have a quiz on fractions", "quiz"),
        ("let's practice", "practice"),
        ("Practice!", "practice"),
        ("I want to practice", "practice"),
        ("I'm done with the quiz", "help"),
        ("Stop quizzing me", "help"),
        ("what is three times four", None),
        ("I practiced yesterday", None),
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
        ("can you repeat the problem?", True),
        ("say that again", True),
        ("what was the question?", True),
        ("Sorry?", True),
        ("I didn't hear it", True),
        ("check my answer", False),
        ("I got x = 5", False),
        ("I'm stuck", False),
    ],
)
def test_is_repeat_request(text: str, expected: bool) -> None:
    assert is_repeat_request(text) is expected


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
        # "what is X?" about a math idea asks to have it explained: a lesson with a worked example
        ("What is a fraction?", "a fraction"),
        ("what are fractions", "fractions"),
        ("What's a common denominator?", "a common denominator"),
        ("what is an equation", "an equation"),
        ("What does numerator mean?", "numerator"),
        # Whisper often hears "what is a ..." as "what is that ...": with an empty board it can only mean the idea
        ("What is that common denominator?", "common denominator"),
        ("what's this fraction", "fraction"),
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
        # about the board: the student's own equation, answer or step
        "how do I solve for x in the equation on the board",  # "board" was cut off at eight words
        "how do I solve the equation",
        "how do I factor the expression",
        "how can I check my answer to the equation",
        "how do I move the 5 to the other side of the equation",
        "Can you explain the equation?",
        "Can you teach me the next step?",
        "teach me how to do the next step",
        "How do I get rid of the fraction?",
        "how do I isolate the variable",
        "how do I simplify the left side of the equation",
        "explain the next step of the equation",
        # an explicit problem: step-by-step help with it, not a lesson
        "how do I solve 3x + 5 = 20",
        "how should I factor x squared plus 5x plus 6",
        "how do I add one half and one third",
        "teach me how to solve three x plus five equals twenty",
        "how do I do this one?",
        "why is this wrong",
        # "what is ..." without a math idea, or about the board: step-by-step help as before
        "what is three plus five",
        "what is 2/3 + 1/4",
        "what is the answer",
        "what is x",
        "what is the next step",
        "what is the denominator of my fraction",
        "what does this mean",
    ],
)
def test_detect_learn_request_ignores_board_questions(text: str) -> None:
    assert detect_learn_request(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "How do I add fractions?",
        "explain fractions to me",
        "how to graph a line",
        "what is a fraction",
        "what is that common denominator",
    ],
)
def test_how_to_forms_are_about_the_board_while_it_is_in_use(text: str) -> None:
    assert detect_learn_request(text) is not None
    assert detect_learn_request(text, board_in_use=True) is None


def test_learn_and_teach_me_still_start_a_lesson_while_the_board_is_in_use() -> None:
    assert detect_learn_request("teach me fractions", board_in_use=True) == "fractions"
    assert detect_learn_request("I want to learn about the order of operations", board_in_use=True) == (
        "the order of operations"
    )


def test_long_topic_without_board_words_is_cut_to_eight_words() -> None:
    topic = detect_learn_request(
        "teach me how to add and subtract fractions with unlike denominators and whole numbers"
    )
    assert topic == "adding and subtract fractions with unlike denominators and"


def test_what_does_mean_stays_fast_on_long_input() -> None:
    import time

    started = time.perf_counter()
    assert detect_learn_request("what does fraction " * 1600) is None
    assert time.perf_counter() - started < 0.5


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
        ("I don't", "generate"),
        ("um I don't", "generate"),
        ("Give me an easy one", "generate"),
        ("give me a hard problem", "generate"),
        ("what is a fraction", None),
        ("I don't get it", None),
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
    assert t.to_dict() == {"mode": "help", "learn_topic": None, "quiz": None, "problem": None}


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


def test_quiz_without_problems_invents_progressive_ones() -> None:
    t = TutorSession(rng=random.Random(7))
    t.start_quiz([])
    assert t.quiz_total == DEFAULT_QUIZ_LENGTH and t.invented
    seen: list[QuizProblem] = []
    for n in range(1, DEFAULT_QUIZ_LENGTH + 1):
        current = t.current()
        assert current is not None and t.quiz_active
        assert t.to_dict()["quiz"]["problem"] == current.problem, "the card shows the invented problem"
        assert current.answer == solve_problem(current.problem) and current.source == INVENTED_SOURCE
        upcoming = t.next_problem()
        assert (upcoming is None) == (n == DEFAULT_QUIZ_LENGTH)
        assert t.current() is current, "the same problem until it is answered"
        seen.append(current)
        t.record(True)
        if upcoming is not None:
            assert t.current() is upcoming, "the problem read out as 'next' is the one checked next"
    assert not t.quiz_active and t.score == DEFAULT_QUIZ_LENGTH
    levels = [p.level for p in seen]
    assert levels == sorted(levels) and levels[0] == 1 and levels[-1] > 3, "easy first, harder later"


def test_invented_quiz_never_repeats_a_problem() -> None:
    # Problems one and two come from the same generator (176 possible): seed 106 drew "x - 9 = 2" twice.
    for seed in (106, *range(300)):
        t = TutorSession(rng=random.Random(seed))
        t.start_quiz([])
        problems: list[str] = []
        while t.quiz_active:
            current = t.current()
            assert current is not None
            problems.append(current.problem)
            t.record(True)
        assert len(set(problems)) == len(problems), f"seed {seed} repeated a problem: {problems}"


def test_invented_quiz_length() -> None:
    t = TutorSession()
    t.start_quiz([], length=3)
    assert t.quiz_total == 3
    t.start_quiz(_problems("1 + 1"), length=3)
    assert t.quiz_total == 1 and not t.invented


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
    assert t.to_dict() == {"mode": "help", "learn_topic": None, "quiz": None, "problem": None}


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
    t.expire_offer()
    assert t.learn_topic == "fractions", "an answered offer keeps its lesson"


def test_unanswered_offer_expires() -> None:
    t = TutorSession()
    t.start_learning("fractions")
    t.expire_offer()
    assert not t.offer_pending and t.learn_topic is None


@pytest.mark.parametrize(("mode", "topic"), [("help", "fractions"), ("practice", None), ("quiz", None)])
def test_mode_change_ends_the_offer(mode: str, topic: str | None) -> None:
    t = TutorSession()
    t.start_learning("fractions")
    t.set_mode(mode)
    assert not t.offer_pending and t.learn_topic == topic


def test_quiz_start_ends_the_lesson() -> None:
    t = TutorSession()
    t.start_learning("fractions")
    t.start_quiz([])
    assert not t.offer_pending and t.learn_topic is None


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


# ---- English and Spanish together ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        ("Estoy atorado, ¿qué hago ahora?", "help"),
        ("Dame una pista", "help"),
        ("Hazme un quiz", "quiz"),
        ("Dame otro parecido", "practice"),
        ("Termina el quiz", "help"),
        ("help me", "help"),
        ("quiz me", "quiz"),
        ("Revisa mi respuesta", None),
        ("ok", None),
    ],
)
@pytest.mark.parametrize("language", ["en", "es"])
def test_detect_mode_any_reads_both_languages(text: str, mode: str | None, language: str) -> None:
    from biobench_sar.tutor import detect_mode_any

    assert detect_mode_any(text, language) == mode


def test_detect_mode_any_tries_the_reply_language_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from biobench_sar import lang, tutor

    monkeypatch.setattr(tutor, "detect_mode", lambda text: "practice")
    monkeypatch.setattr(lang, "detect_mode_es", lambda text: "quiz")
    assert tutor.detect_mode_any("x", "es") == "quiz"
    assert tutor.detect_mode_any("x", "en") == "practice"


def test_spanish_detector_failure_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    from biobench_sar import lang, tutor

    def broken(text: str) -> str | None:
        raise RuntimeError("regex blew up")

    monkeypatch.setattr(lang, "detect_mode_es", broken)
    assert tutor.detect_mode_any("help me", "es") == "help"
    assert tutor.detect_mode_any("dame una pista", "es") is None


@pytest.mark.parametrize(
    ("text", "board_in_use", "topic"),
    [
        ("Quiero aprender a resolver fracciones", False, "resolver fracciones"),
        ("Enséñame ecuaciones", True, "ecuaciones"),  # says it outright: a lesson even with ink on the board
        ("¿Cómo se suman fracciones?", False, "sumar fracciones"),
        ("¿Cómo se suman fracciones?", True, None),  # with ink on the board it is about that ink
        ("¿Cómo resuelvo la ecuación 3x + 5 = 20?", False, None),  # their own problem: help, not a lesson
        ("¿Cómo sumo un medio más un tercio?", False, None),
        ("Explícame la siguiente paso, estoy atorado", False, None),
        ("I want to learn fractions", True, "fractions"),
        ("how do I add fractions?", True, None),
    ],
)
def test_detect_learn_request_any(text: str, board_in_use: bool, topic: str | None) -> None:
    from biobench_sar.tutor import detect_learn_request_any

    assert detect_learn_request_any(text, "es", board_in_use=board_in_use) == topic
    assert detect_learn_request_any(text, "en", board_in_use=board_in_use) == topic


@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("Invéntame uno", "generate"),
        ("No, tú escoge", "generate"),
        ("Tengo uno", "own"),
        ("Yo lo escribo", "own"),
        ("make one up", "generate"),
        ("I have one", "own"),
        ("¿Qué es una fracción?", None),
    ],
)
def test_detect_offer_reply_any(text: str, reply: str | None) -> None:
    from biobench_sar.tutor import detect_offer_reply_any

    assert detect_offer_reply_any(text, "es") == reply
    assert detect_offer_reply_any(text, "en") == reply


@pytest.mark.parametrize(
    ("text", "stop"),
    [("termina el quiz", True), ("ya no quiero hacer el quiz", True), ("stop the quiz", True), ("ayuda", False)],
)
def test_is_quiz_stop_any(text: str, stop: bool) -> None:
    from biobench_sar.tutor import is_quiz_stop_any

    assert is_quiz_stop_any(text) is stop


@pytest.mark.parametrize(
    ("text", "repeat"),
    [
        ("¿Me lo repites?", True),
        ("Repítelo, por favor", True),
        ("¿Puedes repetir el problema?", True),
        ("¿Cuál era el problema?", True),
        ("No escuché", True),
        ("¿Qué?", True),
        ("Dilo otra vez", True),
        ("repeat the problem", True),
        ("Ya lo escribí", False),
        ("¿Qué hago ahora?", False),
    ],
)
def test_is_repeat_request_any(text: str, repeat: bool) -> None:
    from biobench_sar.tutor import is_repeat_request_any

    assert is_repeat_request_any(text) is repeat


# ---- session: the problem on the projector's card -------------------------------------------------------------


def test_show_problem_puts_it_in_the_dict() -> None:
    t = TutorSession()
    t.show_problem("  4x +  7 = 19 ", "dictated")
    assert t.problem == {"text": "4x + 7 = 19", "source": "dictated"}
    assert t.to_dict()["problem"] == {"text": "4x + 7 = 19", "source": "dictated"}


@pytest.mark.parametrize("text", ["", "   ", None])
def test_show_problem_ignores_empty_text(text: str | None) -> None:
    t = TutorSession()
    t.show_problem("2x + 3 = 11", "board")
    t.show_problem(text, "dictated")
    assert t.problem == {"text": "2x + 3 = 11", "source": "board"}


def test_show_problem_caps_long_text_at_a_word() -> None:
    t = TutorSession()
    t.show_problem("word " * 40, "board")
    text = t.problem["text"]
    assert len(text) <= MAX_PROBLEM_CHARS and text.endswith("…") and not text.endswith(" …")


def test_show_problem_rejects_unknown_source() -> None:
    with pytest.raises(ValueError):
        TutorSession().show_problem("x = 1", "guess")


def test_new_topic_and_quiz_clear_the_problem() -> None:
    t = TutorSession()
    t.show_problem("x + 3 = 8", "dictated")
    t.start_learning("fractions")
    assert t.problem is None
    t.show_problem("x + 3 = 8", "dictated")
    t.set_mode("quiz")
    assert t.problem is None


def test_running_quiz_owns_the_card() -> None:
    t = TutorSession()
    t.set_mode("quiz")
    t.start_quiz([], length=3)
    t.show_problem("x + 3 = 8", "board")
    assert t.to_dict()["problem"] is None


def test_a_quiz_started_directly_clears_the_card_and_holds_it() -> None:
    # The dashboard's "Start quiz" calls start_quiz without set_mode: the old problem must not come back after.
    t = TutorSession()
    t.show_problem("4x + 7 = 19", "dictated")
    t.start_quiz([], length=3)
    t.show_problem("2x = 8", "board")  # a help turn that was already running when the quiz started
    t.stop_quiz()
    assert t.problem is None and t.to_dict()["problem"] is None


def test_clear_problem() -> None:
    t = TutorSession()
    t.show_problem("x + 3 = 8", "dictated")
    t.clear_problem()
    assert t.problem is None


def test_quiz_mode_holds_the_card_before_the_quiz_starts() -> None:
    # The dashboard switches to quiz mode; the quiz itself starts on the next question. A practice turn that was
    # already running must not put its problem on the card in between.
    t = TutorSession()
    t.set_mode("practice")
    t.set_mode("quiz")
    t.show_problem("4x + 7 = 19", "dictated")
    assert t.problem is None and t.to_dict()["problem"] is None
