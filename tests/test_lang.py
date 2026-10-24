from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from biobench_sar import lang
from biobench_sar.lang import (
    detect_all,
    detect_language,
    detect_learn_request_es,
    detect_mode_es,
    detect_offer_reply_es,
    is_quiz_stop_es,
    normalize_language,
)


# ---- normalize_language --------------------------------------------------------
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("  ", "auto"),
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("en", "en"),
        ("EN-us", "en"),
        ("en_GB", "en"),
        ("english", "en"),
        ("Inglés", "en"),
        ("es", "es"),
        (" es ", "es"),
        ("es-MX", "es"),
        ("es_419", "es"),
        ("spanish", "es"),
        ("Español", "es"),
        ("fr", "auto"),
        ("klingon", "auto"),
    ],
)
def test_normalize_language(code: str | None, expected: str) -> None:
    assert normalize_language(code) == expected


# ---- detect_language -----------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "default", "expected"),
    [
        # English, short and long
        ("help", "es", "en"),
        ("Help!", "es", "en"),
        ("next step", "es", "en"),
        ("I'm stuck", "es", "en"),
        ("im stuck", "es", "en"),
        ("what's next?", "es", "en"),
        ("quiz me", "es", "en"),
        ("thanks", "es", "en"),
        ("yes", "es", "en"),
        ("can you give me a hint", "es", "en"),
        ("is x equal to five?", "es", "en"),
        ("I don't understand how to solve this equation", "es", "en"),
        ("Teach me how to add fractions with different denominators", "es", "en"),
        ("José wants to know how to add fractions", "es", "en"),
        # Spanish, with and without accents
        ("ayuda", "en", "es"),
        ("¿y ahora?", "en", "es"),
        ("¡ayúdame!", "en", "es"),
        ("no entiendo", "en", "es"),
        ("que sigue", "en", "es"),
        ("¿qué sigue?", "en", "es"),
        ("estoy atorado", "en", "es"),
        ("examíname", "en", "es"),
        ("sí", "en", "es"),
        ("hola", "en", "es"),
        ("gracias", "en", "es"),
        ("tú escoge", "en", "es"),
        ("hazme un quiz", "en", "es"),
        ("dame otro problema parecido", "en", "es"),
        ("como le hago para sumar fracciones", "en", "es"),
        ("¿Cómo se resuelve una ecuación de primer grado?", "en", "es"),
        ("quiero aprender a resolver fracciones", "en", "es"),
        ("x es igual a cinco", "en", "es"),
        ("dos más tres", "en", "es"),
        ("Hola profe, no le entiendo al paso dos, ¿me puede explicar otra vez?", "en", "es"),
        # Spanglish follows the language most of the words are in
        ("can you help me with this, por favor", "es", "en"),
        ("what's the next paso", "es", "en"),
        ("solve for x por favor", "es", "en"),
        ("no entiendo el siguiente step", "en", "es"),
        ("quiero hacer el quiz now", "en", "es"),
        ("ok, ¿y ahora qué hago con la x?", "en", "es"),
        ("examíname please", "en", "es"),
        # math only or nothing either way: the default
        ("3x + 5 = 12", "en", "en"),
        ("3x + 5 = 12", "es", "es"),
        ("x^2 - 4 = 0", "es", "es"),
        ("2/3 + 1/4", "en", "en"),
        ("y = 2x + 1", "es", "es"),
        ("y = 2x + 1", "en", "en"),
        ("3 y 4", "en", "en"),
        ("a + b = c", "es", "es"),
        ("f(x) = 3x", "es", "es"),
        ("√9 = 3", "es", "es"),
        ("25%", "en", "en"),
        ("(2, 3)", "es", "es"),
        ("", "en", "en"),
        ("", "es", "es"),
        ("   ", "es", "es"),
        ("ok", "es", "es"),
        ("ok", "en", "en"),
        ("quiz", "es", "es"),
        ("quiz", "en", "en"),
        ("no", "es", "es"),
        ("no", "en", "en"),
        ("y", "en", "en"),
    ],
)
def test_detect_language(text: str, default: str, expected: str) -> None:
    assert detect_language(text, default) == expected


def test_detect_language_defaults_to_english() -> None:
    assert detect_language("2 + 2") == "en"


@pytest.mark.parametrize(("default", "expected"), [("auto", "en"), ("fr", "en"), ("es-MX", "es"), ("Español", "es")])
def test_detect_language_normalizes_the_default(default: str, expected: str) -> None:
    assert detect_language("x = 4", default) == expected


def test_word_lists_do_not_overlap() -> None:
    assert lang._ES_WORDS.isdisjoint(lang._EN_WORDS)


# ---- detect_mode_es --------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("examíname", "quiz"),
        ("Examíname por favor", "quiz"),
        ("hazme una prueba", "quiz"),
        ("hazme un quiz", "quiz"),
        ("quiero un quiz", "quiz"),
        ("quiero hacer un quiz de fracciones", "quiz"),
        ("¿me haces un quiz?", "quiz"),
        ("dame un examen", "quiz"),
        ("hagamos un quiz", "quiz"),
        ("ponme a prueba", "quiz"),
        ("modo quiz", "quiz"),
        ("termina el quiz", "help"),
        ("para el quiz", "help"),
        ("¡Para el quiz!", "help"),
        ("ya para el quiz", "help"),
        ("oye, termina el quiz por favor", "help"),
        ("terminemos el quiz", "help"),
        ("quiero salir del quiz", "help"),
        ("cancela el examen", "help"),
        ("ya no quiero hacer el quiz", "help"),
        ("no más quiz", "help"),
        ("otro problema parecido", "practice"),
        ("dame uno similar", "practice"),
        ("uno similar", "practice"),
        ("dame otro así", "practice"),
        ("otro igual", "practice"),
        ("un ejercicio parecido", "practice"),
        ("más práctica", "practice"),
        ("necesito más práctica", "practice"),
        ("quiero practicar más", "practice"),
        ("ayuda", "help"),
        ("¡ayúdame!", "help"),
        ("¿me ayudas?", "help"),
        ("el siguiente paso", "help"),
        ("¿cuál es el siguiente paso?", "help"),
        ("no entiendo", "help"),
        ("no le entiendo", "help"),
        ("estoy atorado", "help"),
        ("estoy atorada", "help"),
        ("me atoré", "help"),
        ("¿qué sigue?", "help"),
        ("¿y ahora qué?", "help"),
        ("y ahora", "help"),
        ("¿me das una pista?", "help"),
        ("no sé qué hacer", "help"),
    ],
)
def test_detect_mode_es(text: str, expected: str) -> None:
    assert detect_mode_es(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "¿para qué sirve el quiz?",
        "no quiero un quiz",
        "¿cuándo termina el quiz?",
        "¿cuánto falta para terminar el quiz?",
        "no puedo terminar el quiz",
        "tengo que estudiar para el quiz",
        "el quiz estuvo fácil",
        "¿cuántas preguntas tiene el examen?",
        "no necesito ayuda",
        "no quiero otro problema parecido",
        "dos más uno igual a tres",
        "hola, ¿cómo estás?",
        "x es igual a cinco",
        "quiz",
        "help",
        "",
    ],
)
def test_detect_mode_es_ignores(text: str) -> None:
    assert detect_mode_es(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("termina el quiz", True),
        ("ya no quiero hacer el quiz", True),
        ("ayuda", False),
        ("estoy atorado", False),
        ("¿para qué sirve el quiz?", False),
        ("no puedo terminar el quiz", False),
    ],
)
def test_is_quiz_stop_es(text: str, expected: bool) -> None:
    assert is_quiz_stop_es(text) is expected


# ---- detect_learn_request_es -------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("quiero aprender a resolver fracciones", "resolver fracciones"),
        ("Quiero aprender fracciones", "fracciones"),
        ("quiero aprender más sobre porcentajes", "porcentajes"),
        ("quiero aprender a sumar fracciones por favor", "sumar fracciones"),
        ("me gustaría aprender álgebra", "álgebra"),
        ("quiero aprender cómo se calcula el promedio", "calcular el promedio"),
        ("enséñame ecuaciones", "ecuaciones"),
        ("Enséñame a dividir fracciones, por favor", "dividir fracciones"),
        ("¿me enseñas a despejar una variable?", "despejar una variable"),
        ("¿me puedes enseñar la regla de tres?", "la regla de tres"),
        ("¿cómo se suman fracciones?", "sumar fracciones"),
        ("como se suman fracciones", "sumar fracciones"),
        ("¿Cómo se resuelven las ecuaciones de primer grado?", "resolver las ecuaciones de primer grado"),
        ("¿cómo puedo sacar el área de un triángulo?", "sacar el área de un triángulo"),
        ("¿cómo le hago para multiplicar fracciones?", "multiplicar fracciones"),
        ("¿cómo resuelvo ecuaciones con fracciones?", "resolver ecuaciones con fracciones"),
        ("explícame las fracciones equivalentes", "las fracciones equivalentes"),
        ("explícame cómo se multiplican números negativos", "multiplicar números negativos"),
    ],
)
def test_detect_learn_request_es(text: str, expected: str) -> None:
    assert detect_learn_request_es(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "¿cómo resuelvo esto?",
        "¿cómo resuelvo esta ecuación?",
        "¿cómo se suma 2/3 + 1/4?",
        "enséñame esto",
        "enséñame a hacerlo",
        "explícame este paso",
        "explícame otra vez",
        "no quiero aprender fracciones",
        "ya aprendí fracciones",
        "¿cómo estás?",
        "¿cómo se dice fracción en inglés?",
        "¿cómo le hago?",
        "hola",
        "teach me fractions",
        "",
    ],
)
def test_detect_learn_request_es_ignores(text: str) -> None:
    assert detect_learn_request_es(text) is None


def test_learn_topic_is_capped() -> None:
    topic = detect_learn_request_es(
        "quiero aprender a resolver ecuaciones de primer grado con fracciones y decimales juntos"
    )
    assert topic is not None
    assert len(topic.split()) == lang.MAX_TOPIC_WORDS


# ---- detect_offer_reply_es ---------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("invéntame uno", "generate"),
        ("Invéntame uno, porfa", "generate"),
        ("dame uno", "generate"),
        ("hazme uno", "generate"),
        ("hazme uno fácil", "generate"),
        ("¿me das uno?", "generate"),
        ("tú escoge", "generate"),
        ("escoge tú", "generate"),
        ("como quieras", "generate"),
        ("sorpréndeme", "generate"),
        ("no tengo", "generate"),
        ("no tengo ninguno", "generate"),
        ("no", "generate"),
        ("no, gracias", "generate"),
        ("nel", "generate"),
        ("no, dame uno tú", "generate"),
        ("tengo uno", "own"),
        ("Sí, tengo uno", "own"),
        ("sí tengo", "own"),
        ("ya tengo mi ejercicio", "own"),
        ("tengo uno de mi tarea", "own"),
        ("voy a escribir el mío", "own"),
        ("lo voy a escribir", "own"),
        ("déjame escribirlo", "own"),
        ("yo lo escribo", "own"),
        ("el mío", "own"),
        ("No, tengo uno", "own"),
    ],
)
def test_detect_offer_reply_es(text: str, expected: str) -> None:
    assert detect_offer_reply_es(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "hola",
        "sí",
        "¿qué?",
        "no sé",
        "dame un momento",
        "dame una pista",
        "tengo una duda",
        "tengo una pregunta",
        "tengo hambre",
        "no entiendo",
        "make one up",
        "",
    ],
)
def test_detect_offer_reply_es_ignores(text: str) -> None:
    assert detect_offer_reply_es(text) is None


# ---- detect_all ------------------------------------------------------------------
@pytest.fixture
def fresh_english() -> Iterator[None]:
    lang._english_detectors.cache_clear()
    yield
    lang._english_detectors.cache_clear()


def _fake_tutor(mode: str | None = None, topic: str | None = None, offer: str | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        detect_mode=lambda text: mode if "quiz me" in text or mode == "always" else None,
        detect_learn_request=lambda text: topic if "learn" in text else None,
        detect_offer_reply=lambda text: offer if "make one" in text else None,
    )


def test_detect_all_reads_spanish(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", None)
    assert detect_all("examíname") == {"language": "es", "mode": "quiz", "learn_topic": None, "offer_reply": None}
    assert detect_all("enséñame ecuaciones")["learn_topic"] == "ecuaciones"
    assert detect_all("invéntame uno")["offer_reply"] == "generate"


def test_detect_all_without_tutor_module(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", None)
    assert lang._english_detectors() is None
    assert detect_all("quiz me") == {"language": "en", "mode": None, "learn_topic": None, "offer_reply": None}


def test_detect_all_uses_english_detectors(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", _fake_tutor("quiz", "fractions", "generate"))
    assert detect_all("quiz me") == {"language": "en", "mode": "quiz", "learn_topic": None, "offer_reply": None}
    assert detect_all("I want to learn fractions")["learn_topic"] == "fractions"
    assert detect_all("make one up")["offer_reply"] == "generate"


def test_detect_all_reads_spanglish_both_ways(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", _fake_tutor("quiz"))
    found = detect_all("ok quiz me, y luego enséñame ecuaciones")
    assert found["language"] == "es"
    assert (found["mode"], found["learn_topic"]) == ("quiz", "ecuaciones")


def test_detect_all_prefers_the_detected_language(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", _fake_tutor("always"))
    assert detect_all("examíname")["mode"] == "quiz"
    assert detect_all("I need help")["mode"] == "always"


def test_detect_all_ignores_a_tutor_without_detectors(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", types.SimpleNamespace(detect_mode=lambda text: "quiz"))
    assert lang._english_detectors() is None


def test_detect_all_default_breaks_ties(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", None)
    assert detect_all("x = 5")["language"] == "en"
    assert detect_all("x = 5", default="es")["language"] == "es"


def test_detect_all_with_real_tutor(fresh_english: None) -> None:
    tutor = pytest.importorskip("biobench_sar.tutor")
    if not hasattr(tutor, "detect_mode"):
        pytest.skip("tutor has no English detectors")
    assert detect_all("quiz me")["mode"] == "quiz"
    assert detect_all("examíname")["mode"] == "quiz"
