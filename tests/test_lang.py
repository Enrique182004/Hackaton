from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import time
import types
from collections.abc import Callable, Iterator

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
        # display names, as a voice or locale picker shows them
        ("Spanish (Mexico)", "es"),
        ("Español (México)", "es"),
        ("English (United States)", "en"),
        ("(es)", "es"),
        ("fr", "auto"),
        ("klingon", "auto"),
        ("French (France)", "auto"),
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
        # short replies and slang a Mexican student types without accents
        ("nel", "en", "es"),
        ("sip", "en", "es"),
        ("simón", "en", "es"),
        ("simon", "en", "es"),
        ("a ver", "en", "es"),
        ("espera", "en", "es"),
        ("esperame tantito", "en", "es"),
        ("no me sale", "en", "es"),
        ("ocupo ayuda", "en", "es"),
        ("ando atorado", "en", "es"),
        ("neta no le entiendo", "en", "es"),
        ("q onda profe, como le hago?", "en", "es"),
        ("because", "es", "en"),
        ("because I don't get it", "es", "en"),
        ("dale", "en", "es"),
        ("nop", "en", "es"),
        ("no mames", "en", "es"),
        ("practiquemos", "en", "es"),
        ("toy atorao", "en", "es"),
        ("me neither", "es", "en"),
        ("no me neither", "en", "en"),
        # short answers: the verbs and units of a word problem
        ("me sale 7 y sobra 1", "en", "es"),
        ("da 2.5 litros", "en", "es"),
        ("cuesta 20 pesos", "en", "es"),
        # "pesos" / "kilos" are English words too, so alone they vote for neither
        ("5 kilos", "en", "en"),
        ("20 pesos", "en", "en"),
        ("20 pesos", "es", "es"),
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
        # feminine "la prueba" / "la evaluación", and how students really type it
        ("hazme la prueba", "quiz"),
        ("empecemos la prueba", "quiz"),
        ("inicia la prueba", "quiz"),
        ("dame la prueba", "quiz"),
        ("quiero hacer la prueba", "quiz"),
        # with "haz / hagamos" a bare "la prueba" is checking the answer; a topic makes it a quiz
        ("hagamos la prueba de fracciones", "quiz"),
        ("vamos a hacer la prueba de fracciones", "quiz"),
        ("hagamos la prueba ya, pregúntame", "quiz"),
        ("hazme una pruebita", "quiz"),
        ("dame la pruebita", "quiz"),
        ("aviéntame un examen", "quiz"),
        ("¿me tomas la lección?", "quiz"),
        ("quiero que me preguntes", "quiz"),
        ("otro quiz pero más difícil", "quiz"),
        ("¿por qué no me puedes hacer un quiz?", "quiz"),
        ("luego luego hazme un quiz", "quiz"),
        ("aplicame la evaluacion", "quiz"),
        ("ponme un examen", "quiz"),
        ("hazme un quiz porfa", "quiz"),
        ("me haces un examen?", "quiz"),
        ("¿podemos hacer un quiz?", "quiz"),
        ("puedes hacerme un quiz", "quiz"),
        ("hay que hacer un quiz", "quiz"),
        ("pregúntame", "quiz"),
        ("preguntame", "quiz"),
        ("hazme preguntas", "quiz"),
        ("hazme unas preguntas de fracciones", "quiz"),
        ("hazme cinco preguntas", "quiz"),
        ("otro quiz", "quiz"),
        ("ok, otro quiz porfa", "quiz"),
        ("otra prueba", "quiz"),
        ("¿por qué no me haces un quiz?", "quiz"),
        ("por que no hacemos un examen", "quiz"),
        ("vamos al quiz", "quiz"),
        ("tómame la lección", "quiz"),
        ("tomame la leccion", "quiz"),
        ("termina el quiz", "help"),
        ("termina la prueba", "help"),
        ("terminemos esta prueba", "help"),
        ("cancela la evaluación", "help"),
        ("cancela la evaluacion", "help"),
        ("ya no quiero la prueba", "help"),
        ("sal de la prueba", "help"),
        ("ya párale al quiz", "help"),
        ("parale al examen", "help"),
        ("ya parale a la prueba", "help"),
        ("para esta evaluación", "help"),
        ("para la prueba", "help"),
        ("para el quiz, porfa", "help"),
        ("termina la prueba de fracciones", "help"),
        ("termina el quiz, ya me cansé", "help"),
        ("no quiero seguir con el quiz", "help"),
        ("ya no quiero seguir con la prueba", "help"),
        ("no mas pruebas", "help"),
        ("basta de la prueba", "help"),
        ("ya me canse del quiz", "help"),
        ("ya estuvo con el examen", "help"),
        ("quiero salirme de la prueba", "help"),
        ("salte del quiz", "help"),
        ("para el quiz", "help"),
        ("¡Para el quiz!", "help"),
        ("ya para el quiz", "help"),
        ("oye, termina el quiz por favor", "help"),
        ("terminemos el quiz", "help"),
        ("quiero salir del quiz", "help"),
        ("cancela el examen", "help"),
        ("ya no quiero hacer el quiz", "help"),
        ("no más quiz", "help"),
        # a stop followed by more words, typed without commas
        ("termina el quiz ya me canse", "help"),
        ("termina el quiz ya me cansé", "help"),
        ("termina la prueba ya me canse", "help"),
        ("termina el quiz y ya", "help"),
        ("termina el quiz que ya me aburri", "help"),
        ("acaba el quiz que tengo sueño", "help"),
        ("termina el examen ya estuvo", "help"),
        ("termina el examen por favor", "help"),
        ("termina el quiz por hoy", "help"),
        ("para el quiz ya no quiero", "help"),
        ("para el quiz ya no quiero seguir", "help"),
        ("para el quiz no quiero seguir", "help"),
        ("para el quiz, ya entendí", "help"),
        ("para el quiz porque ya me cansé", "help"),
        ("para el quiz y ya", "help"),
        ("para el quiz pls", "help"),
        ("para el quiz please", "help"),
        ("ya para el quiz, gracias", "help"),
        # how students open a sentence
        ("órale, termina el quiz", "help"),
        ("ándale, ya párale al examen", "help"),
        ("porfis termina la prueba", "help"),
        ("sí, termina el quiz", "help"),
        ("entonces termina la prueba", "help"),
        ("ay ya termina el examen", "help"),
        ("para, para el quiz", "help"),
        # more ways to quit
        ("ya no más preguntas", "help"),
        ("ya no quiero más preguntas", "help"),
        ("sácame del quiz", "help"),
        ("sácame de la prueba", "help"),
        ("¿cómo salgo del examen?", "help"),
        ("no sé cómo salir del quiz", "help"),
        ("mejor dejamos el examen para mañana", "help"),
        ("ya me aburrió el quiz", "help"),
        ("suficiente quiz por hoy", "help"),
        ("ya estuvo, termina el quiz", "help"),
        ("mejor ya no le sigo al quiz", "help"),
        ("ya no le sigo", "help"),
        ("ya no quiero contestar más", "help"),
        ("¿cómo se sale del quiz?", "help"),
        ("¿cómo dejo el examen?", "help"),
        ("ya estuvo de preguntas", "help"),
        ("basta de preguntas", "help"),
        ("ya dejamos el examen", "help"),
        ("suficiente prueba por hoy", "help"),
        # a quit with more after it: a new clause, a topic, or "hasta aquí" ("let's stop here")
        ("ya termina el quiz, mañana sigo", "help"),
        ("termina el examen, mañana le seguimos", "help"),
        ("termina el quiz, hoy ya no quiero", "help"),
        ("termina el quiz, el lunes lo acabamos", "help"),
        ("termina el quiz, mañana le seguimos", "help"),
        ("termina el examen, hoy ya no puedo", "help"),
        ("termina el quiz, a las tres tengo clase", "help"),
        ("termina el quiz, cuando quieras", "help"),
        ("ya termina el quiz, el viernes lo seguimos", "help"),
        ("Termina el quiz. Mañana le seguimos.", "help"),
        ("Termina el quiz. Hasta aquí.", "help"),
        ("Termina el quiz. Hoy no puedo.", "help"),
        ("termina el quiz, hasta aquí", "help"),
        ("ya me aburrió el quiz, mañana sigo", "help"),
        ("ya termina el quiz mañana sigo", "help"),
        ("termina el quiz hoy ya no quiero", "help"),
        ("termina el quiz el lunes lo acabamos", "help"),
        ("ya acaba el examen de los decimales porfa", "help"),
        ("termina el quiz de la tabla del 7", "help"),
        ("termina el examen de la unidad 2", "help"),
        ("termina la prueba de las fracciones", "help"),
        ("párale al quiz de las ecuaciones", "help"),
        ("termina el quiz de hoy", "help"),
        ("termina el quiz hasta aquí", "help"),
        ("acaba el examen hasta aquí", "help"),
        ("termina el quiz luego luego", "help"),
        ("Sí termina el quiz", "help"),
        ("no, para el quiz, ya no quiero", "help"),
        ("para el quiz ya me cansé de las preguntas", "help"),
        # the reason first, then the command in its own clause
        ("ya me cansé, termina el quiz", "help"),
        ("no sé, termina el quiz", "help"),
        ("ya no puedo, termina el examen", "help"),
        ("está muy difícil, termina el quiz", "help"),
        ("esta bien dificil, ya termina el quiz porfa", "help"),
        ("no sé, mejor párale al quiz", "help"),
        ("ya me aburrí, para el quiz", "help"),
        ("uff, termina el quiz", "help"),
        ("neta, termina el quiz", "help"),
        ("estoy atorado, termina el quiz", "help"),
        ("ya no le entiendo, mejor termina el quiz", "help"),
        ("no puedo más así que termina el examen", "help"),
        ("ya me cansé y ya termina el quiz", "help"),
        ("está difícil pero termina el quiz", "help"),
        ("ya no puedo. Termina el quiz.", "help"),
        ("ya no puedo. Para el quiz.", "help"),
        ("no sé, para el quiz ya me cansé", "help"),
        # the reason first with no comma: kids and Whisper drop it
        ("ya me cansé termina el quiz", "help"),
        ("ya me canse termina el quiz", "help"),
        ("ya no puedo termina el quiz", "help"),
        ("ya no quiero termina el examen", "help"),
        ("ya me aburrí para el quiz", "help"),
        ("ya me canse terminale al quiz", "help"),
        ("profe ya me cansé ya terminale al quiz", "help"),
        ("profe, ya me cansé termina el quiz", "help"),
        ("ya no puedo más termina el quiz", "help"),
        ("ya me harté párale al quiz", "help"),
        # polite or firm endings, and a time word followed by the student's next clause
        ("termina el quiz si se puede", "help"),
        ("termina el quiz si puedes", "help"),
        ("termina el quiz si no te importa", "help"),
        ("termina el quiz sí o sí", "help"),
        ("termina el quiz luego vemos", "help"),
        ("termina el quiz después vemos", "help"),
        ("termina el quiz mañana hacemos otro", "help"),
        ("termina el quiz mañana te explico", "help"),
        # after "no," a "ya" or "porfa" makes a bare "para el quiz" an order; "ya no," turns the quiz down
        ("no, ya para el quiz", "help"),
        ("no, para el quiz porfa", "help"),
        ("ya no, para el quiz", "help"),
        ("para el quiz, ya no quiero hacerlo", "help"),
        ("para el quiz ya no quiero hacer nada", "help"),
        ("para el quiz que ya no quiero jugar", "help"),
        ("para el quiz ya me cansé mucho", "help"),
        ("ahí le dejamos al quiz", "help"),
        ("ya le paramos al quiz", "help"),
        ("ya terminamos el quiz por hoy", "help"),
        ("mejor dejamos el quiz y repasamos fracciones", "help"),
        ("ya dejamos el examen porque me cansé", "help"),
        ("mejor dejamos el examen y me explicas", "help"),
        ("ya dejamos el examen ahí", "help"),
        ("otro problema parecido", "practice"),
        ("dame uno similar", "practice"),
        ("uno similar", "practice"),
        ("dame otro así", "practice"),
        ("otro igual", "practice"),
        ("un ejercicio parecido", "practice"),
        ("más práctica", "practice"),
        ("necesito más práctica", "practice"),
        ("quiero practicar más", "practice"),
        ("quiero practicar mas", "practice"),
        ("más ejercicios", "practice"),
        ("mas ejercicios", "practice"),
        ("dame mas ejercicios", "practice"),
        ("dame otro ejercicio", "practice"),
        ("otro ejercicio porfa", "practice"),
        ("otro ejercicio más", "practice"),
        ("más problemas de fracciones", "practice"),
        ("quiero hacer mas ejercicios", "practice"),
        ("quiero seguir practicando", "practice"),
        ("dame otro igual a este", "practice"),
        ("otro igualito", "practice"),
        ("otra igual a la anterior", "practice"),
        ("dame otra igual a la tres", "practice"),
        ("otra igual a la que hicimos", "practice"),
        ("practiquemos", "practice"),
        ("practiquemos fracciones", "practice"),
        ("vamos a practicar", "practice"),
        ("hay que practicar", "practice"),
        ("dame otro más difícil", "practice"),
        ("dame otro ejercicio más difícil", "practice"),
        ("uno más difícil porfa", "practice"),
        ("ponme ejercicios de fracciones", "practice"),
        ("dame unos ejercicios", "practice"),
        ("dame problemas para practicar", "practice"),
        ("sigamos con otro ejercicio parecido", "practice"),
        ("vamos con otro de práctica", "practice"),
        ("¿hay otro así?", "practice"),
        ("¿hay otro problema así?", "practice"),
        ("¿hay otro ejercicio parecido?", "practice"),
        ("¿hay otro así porfa?", "practice"),
        ("otro igual a la segunda", "practice"),
        ("dame otra ecuación más difícil", "practice"),
        ("ok, vamos a practicar", "practice"),
        ("ahora sí vamos a practicar", "practice"),
        ("Sí vamos a practicar", "practice"),
        ("simón vamos a practicar", "practice"),
        ("claro vamos a practicar", "practice"),
        ("primero vamos a practicar", "practice"),
        ("mira, ¿hay otro igual?", "practice"),
        # "usted", as a student asks the teacher
        ("¿me da otro igual?", "practice"),
        ("me da otro igual porfa", "practice"),
        ("¿nos da otro igual?", "practice"),
        ("¿me da otro ejercicio parecido?", "practice"),
        ("profe, ¿me da otro problema así?", "practice"),
        ("maestra me da otro ejercicio parecido", "practice"),
        ("¿me da otro parecido, maestra?", "practice"),
        ("practiquemos luego luego", "practice"),
        # "luego" before an order is "then", not "later"
        ("luego dame otro ejercicio", "practice"),
        ("y luego hazme un quiz", "quiz"),
        ("primero explícame y luego hazme un quiz", "quiz"),
        ("explícame fracciones y luego me haces un quiz", "quiz"),
        ("luego hazme un quiz", "quiz"),
        ("hazme un quiz luego luego", "quiz"),
        ("hazme un quiz y luego me explicas", "quiz"),
        ("para el examen de mañana hazme un quiz", "quiz"),
        # a later-word after the request is not read: it may belong to the student's reason
        ("hazme un quiz porque tengo examen mañana", "quiz"),
        ("hazme un quiz porque tengo examen mañana.", "quiz"),
        ("hazme un quiz que tengo examen mañana", "quiz"),
        ("quiero un quiz porque el examen es mañana", "quiz"),
        ("examíname que el examen es mañana", "quiz"),
        ("pregúntame que tengo examen mañana", "quiz"),
        ("hazme un quiz que tengo prueba pasado mañana", "quiz"),
        ("hazme un quiz porfa que el parcial es mañana", "quiz"),
        ("hazme un quiz rápido que me voy al rato", "quiz"),
        ("hazme un quiz que tengo examen más tarde", "quiz"),
        ("hazme un quiz de lo que vimos esta mañana", "quiz"),
        ("hazme un quiz de lo que vamos a ver mañana", "quiz"),
        ("hazme un quiz para el examen de mañana", "quiz"),
        ("tengo examen mañana hazme un quiz", "quiz"),
        ("mañana tengo examen hazme un quiz", "quiz"),
        ("el examen es mañana hazme un quiz", "quiz"),
        ("pero mañana no, hazme un quiz", "quiz"),
        ("hazme un quiz luego", "quiz"),
        ("hazme un quiz después", "quiz"),
        ("quiz de repaso para mañana", "quiz"),
        ("un quiz de repaso para mañana", "quiz"),
        ("Sí otro quiz", "quiz"),
        ("quiero practicar porque tengo examen mañana", "practice"),
        ("quiero practicar fracciones porque el examen es mañana", "practice"),
        ("practiquemos que el examen es mañana", "practice"),
        ("dame más ejercicios que tengo examen mañana", "practice"),
        ("dame otro ejercicio que tengo clase después", "practice"),
        ("dame otro ejercicio como el de la mañana", "practice"),
        ("practiquemos lo que vimos en la mañana", "practice"),
        ("dame otro ejercicio después", "practice"),
        ("practiquemos luego", "practice"),
        ("practiquemos más tarde", "practice"),
        ("practiquemos otro día", "practice"),
        ("quiero practicar más tarde", "practice"),
        ("quiero practicar hoy", "practice"),
        ("quiero practicar hasta que me salga bien", "practice"),
        ("necesito practicar más hasta entenderle", "practice"),
        ("ayuda", "help"),
        ("¡ayúdame!", "help"),
        ("¿me ayudas?", "help"),
        ("¿me puedes ayudar?", "help"),
        ("¿me podrías ayudar?", "help"),
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
        ("ando atorado", "help"),
        ("toy atorado", "help"),
        ("atorado", "help"),
        ("atorada", "help"),
        ("estoy bien perdido", "help"),
        ("no le se", "help"),
        ("no le sé", "help"),
        ("no le agarro la onda", "help"),
        ("¿cómo le hago?", "help"),
        ("como le hago", "help"),
        ("no me sale", "help"),
        ("no me sale el resultado", "help"),
        ("échame la mano", "help"),
        ("¿me echas la mano?", "help"),
        ("hechame una mano", "help"),
        ("no me queda claro", "help"),
        ("ayudame porfa", "help"),
        ("neta no le entiendo", "help"),
        ("no estoy atorado, pero no entiendo el paso dos", "help"),
        ("no entiendo el otro ejercicio", "help"),
        ("no estoy entendiendo", "help"),
        ("no le estoy entendiendo nada", "help"),
        ("sigo sin entender", "help"),
        ("no sé hacerlo", "help"),
        ("no sé resolverlo", "help"),
        ("¿cómo se hace?", "help"),
        ("¿cómo se hace esto?", "help"),
        ("no me acuerdo cómo se hace", "help"),
        ("me bloqueé", "help"),
        ("estoy en blanco", "help"),
        ("una ayudadita", "help"),
        ("¿me explicas?", "help"),
        ("explícame otra vez", "help"),
        ("toy atorao", "help"),
        ("estoy atorao", "help"),
        ("q hago", "help"),
        ("k hago", "help"),
        ("estoy tan atorado", "help"),
        ("atorado con este paso", "help"),
        ("sin tu ayuda no puedo", "help"),
        # "¿cómo le hago para salir del quiz?" quits rather than asking for a hint
        ("¿cómo le hago para salir del quiz?", "help"),
        # stuck, the way Mexican students say it
        ("este problema me tiene atorado", "help"),
        ("este ejercicio me tiene bien confundida", "help"),
        ("me tienes confundido", "help"),
        ("me dejaste más confundido", "help"),
        ("me dejó confundida", "help"),
        ("otra vez atorado", "help"),
        ("otra vez atorada", "help"),
        ("estoy todavía atorado", "help"),
        ("estoy como atorado", "help"),
        ("estoy re confundido", "help"),
        ("estoy demasiado confundido", "help"),
        ("neta atorado", "help"),
        ("no sé sacarlo", "help"),
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
        # "para el examen ..." opening a question is "for the exam", not "stop the exam"
        "para el examen de mañana, ¿qué temas vienen?",
        "para el quiz de manana tengo que estudiar fracciones",
        "para el examen, ¿qué temas vienen?",
        "para la prueba necesito calculadora?",
        "termina el quiz de mañana a las tres",
        "termina el quiz a las tres",
        "termina el quiz el viernes",
        "si termina el quiz me avisas",
        "para el quiz que tengo mañana",
        "para el examen ya estudié",
        "para el examen no quiero sacar seis",
        "para el quiz terminé el tema 3",
        "para el quiz me voy a preparar",
        "para el quiz de mañana ...",
        "dejamos el examen en la mesa",
        "me cansó el quiz de ayer",
        "eso es suficiente prueba de que funciona",
        # "hacer la prueba" is also checking an answer. A bare "hagamos la prueba" right after solving means
        # "let's check it", so it waits for a topic ("de fracciones") or a quiz cue ("pregúntame").
        "hagamos la prueba con x = 2",
        "haz la prueba sustituyendo",
        "vamos a hacer la prueba para ver si está bien",
        "hagamos la prueba",
        "¿hacemos la prueba?",
        "vamos a hacer la prueba",
        "haz la prueba",
        "hagamos la prueba del nueve",
        "hagamos la prueba otra vez",
        # dictation and talk about the worksheet
        "otro igual a cinco",
        "es otro igual a cinco",
        "otro igual a x",
        "otra igual a la mitad",
        "es otra fracción igual",
        "hay otro problema así en la hoja",
        "sigamos con otro tema",
        "practicar cansa",
        "uno más",
        "hay más ejercicios en la hoja",
        "tengo mas problemas con las fracciones",
        "el otro quiz estuvo fácil",
        "encuentra el número perdido",
        # the student asking, or turning one down
        "otra pregunta",
        "dame otro ejemplo",
        "tengo una pregunta",
        "te hago una pregunta",
        "no quiero otro quiz",
        "no quiero más ejercicios",
        "no podemos hacer un quiz",
        "¿por qué hacemos un quiz?",
        "no estoy atorado",
        "ya no estoy atorada",
        "no quiero que me hagas un examen",
        "después me haces un quiz",
        "por que no quiero hacer un quiz",
        "no, por que no quiero hacer un quiz",
        "yo puedo sin ayuda",
        # "atorado" about something else, or negated further back
        "el problema no esta atorado",
        "no estoy tan atorado",
        "la variable esta confundida con y",
        "el carro está atorado",
        "no se hace así",
        "¿cómo se hace una gráfica?",
        "¡así es como se hace!",
        "no me tiene confundido",
        "no se sacaron bien los valores",
        # a time right after the quiz word says when it ends
        "termina el quiz del viernes",
        "termina el quiz de hoy a las tres",
        "termina el quiz mañana a las tres",
        "termina el quiz luego",
        "termina el quiz en un rato",
        "termina el quiz hasta el viernes",
        "termina el examen en una hora",
        "termina el examen en media hora",
        # yes/no questions about the quiz, in the command's word order
        "¿termina el quiz si fallo una?",
        "¿termina el examen solo o le pico algo?",
        "¿termina la prueba con la pregunta 5?",
        # "for the exam": an answer, or a reason that goes on
        "sí, para el examen",
        "no, para el examen",
        "para el examen ya no quiero sacar seis",
        "para el examen ya estoy cansado de estudiar",
        "para el quiz ya me cansé de estudiar",
        "para el quiz ya terminé de estudiar",
        "para el examen ya acabé la guía",
        "para el quiz ya entendí las fracciones",
        "para el quiz ya terminé el tema 3",
        "para el quiz ya no puedo con las fracciones",
        "para el quiz basta con estudiar",
        "eso es suficiente prueba",
        "suficiente prueba",
        "ya dejamos el examen en la mesa",
        "¿cómo salgo del examen con buena calificación?",
        # mid-quiz "sigamos con otro" is "next": practice mode would end the quiz
        "sigamos con otro",
        "sigamos con otro ejercicio",
        "vamos con otro",
        "seguimos con otro, porfa",
        "continuemos con otro",
        "sale, vamos con otro",
        "mejor vamos con otro",
        "no sé, vamos con otro",
        "sáltatelo, vamos con otro",
        "vamos con otro problema",
        # and "otra" / "una" alone is "pregunta"
        "dame otra más fácil",
        "una más difícil",
        "otra más difícil",
        # "el otro igual" describes; "a la tercera parte" / "a los dos" is dictation
        "una fracción es 1/2 y la otra igual a la primera",
        "el otro igual a la segunda",
        "uno es 5 y el otro igual a lo que falta",
        "un lado mide 5 y el otro igual",
        "un ángulo mide 40 y el otro igual al anterior",
        "otra igual a la tercera parte",
        "otra igual a la cuarta parte",
        "otro igual a los dos",
        "¿hay otro así en la hoja?",
        # put off: only a word before the request is read, anywhere in its clause
        "mañana vamos a practicar",
        "en la clase vamos a practicar fracciones",
        "si vamos a practicar 2 horas diarias, ¿cuántas horas son en una semana?",
        "luego me haces un quiz",
        "mañana me haces un quiz",
        "mañana hazme un quiz",
        "ok, mañana hazme un quiz",
        "después hazme un quiz",
        "pero mañana hazme un quiz",
        "pero mañana me haces un quiz",
        "bueno pero mañana hazme un quiz",
        "y mañana hazme un quiz",
        "pero pasado mañana me haces un quiz",
        "tal vez mañana me haces un quiz",
        "si quieres mañana me haces un quiz",
        "o sea mañana me haces un quiz",
        "igual mañana me haces un quiz",
        "mañana en la mañana hazme un quiz",
        "en la mañana hazme un quiz",
        "mañana temprano hazme un quiz",
        "mañana en la tarde me haces un quiz",
        "mañana antes del examen hazme un quiz",
        "pero mañana practiquemos",
        "mañana dame otro igual",
        "luego en la tarde me haces un quiz",
        # the worksheet or a result: "hay otro así" asks only as a question ("¿hay otro así?")
        "aquí hay otro así",
        "mira hay otro igual",
        "hay otro igual",
        "ahí hay otro igual",
        "sale otro igual",
        "queda otro igual",
        "me queda otra igual a la de arriba",
        "pongo otro igual abajo",
        "eso da otro igual",
        "que da otro igual",
        # a condition says when to stop, not to stop now
        "si fallo, termina el quiz",
        "cuando acabe, termina el quiz",
        "hasta que acabe, termina el quiz",
        "en cuanto acabe, termina el quiz",
        "si fallo entonces termina el quiz",
        "¿y entonces termina el quiz?",
        "y si fallo, termina el quiz",
        "pero si fallo, termina el quiz",
        "oye y si fallo, termina el quiz",
        "pero en cuanto acabe, termina el quiz",
        "¿y si fallo, termina el quiz?",
        "¿y cuando acabe, termina el quiz?",
        "¿qué pasa si me equivoco, termina el quiz?",
        "¿Y si contesto mal? ¿Termina el quiz?",
        # a later "termina el quiz" in a question is "the quiz ends"
        "falta una y ya termina el quiz?",
        "¿esta es la última y ya termina el quiz?",
        "me falta una y ya termina el examen verdad?",
        "es la ultima y ya termina el quiz o no",
        "falta una, ¿y ya termina el quiz?",
        "oye pero termina el quiz?",
        "contesto esta entonces termina el quiz?",
        "ya acabé, ¿termina el quiz?",
        "¿a qué hora entonces termina el quiz?",
        "ya me cansé termina el quiz?",
        # after a bare yes, "para el examen" answers a question; "claro, termina el quiz" may agree that it ends
        "sí, para el examen porfa",
        "Sí para el examen porfa",
        "sí, para el quiz por favor",
        "sí, para el examen ya",
        "claro, termina el quiz",
        # a bare "para el quiz" after another clause or a bare "no" is "for the quiz"
        "quiero estudiar, para el examen",
        "tengo que estudiar, para el quiz",
        "no para el quiz",
        "no no, para el quiz",
        "oye no, para el examen",
        "ya me cansé, para el quiz, mañana sigo",
        "ya dejamos el examen aquí en la mesa",
        "ya dejamos el examen ahí en la mesa",
        "mejor dejamos el examen en la mesa",
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
        ("termina la prueba", True),
        ("cancela la evaluación", True),
        ("ya párale al quiz", True),
        ("ya párale a la prueba", True),
        ("para esta evaluacion", True),
        ("para el quiz de mañana, ¿qué temas vienen?", False),
        ("para el examen de mañana, ¿qué temas vienen?", False),
        ("para el quiz de manana tengo que estudiar fracciones", False),
        ("termina el quiz ya me cansé", True),
        ("para el quiz ya no quiero seguir", True),
        ("¿cómo le hago para salir del quiz?", True),
        ("no quiero un quiz", False),
        ("termina el quiz a las tres", False),
        ("¿cómo le hago para terminar el quiz a tiempo?", False),
        ("ya termina el quiz, mañana sigo", True),
        ("termina el quiz de la tabla del 7", True),
        ("termina el quiz hasta aquí", True),
        ("Sí termina el quiz", True),
        ("ya no le sigo", True),
        ("termina el quiz de mañana a las tres", False),
        ("termina el quiz el viernes", False),
        ("termina el quiz del viernes", False),
        ("¿termina el quiz si fallo una?", False),
        ("si termina el quiz me avisas", False),
        ("sí, para el examen", False),
        ("para el quiz ya terminé el tema 3", False),
        ("ya me cansé, termina el quiz", True),
        ("estoy atorado, termina el quiz", True),
        ("ya no le entiendo, mejor termina el quiz", True),
        ("no le entiendo; termina el quiz", True),
        ("no puedo más así que termina el examen", True),
        ("ya me aburrí, para el quiz", True),
        ("termina el quiz si puedes", True),
        ("no, ya para el quiz", True),
        ("si fallo, termina el quiz", False),
        ("cuando acabe, termina el quiz", False),
        ("quiero estudiar, para el examen", False),
        ("no, para el examen", False),
        ("no para el quiz", False),
        ("termina el quiz luego", False),
        ("ya me cansé termina el quiz", True),
        ("ya me aburri para el quiz", True),
        ("ya me cansé de estudiar para el examen", False),
        ("ya me cansé termina el quiz mañana", False),
        ("¿termina el quiz?", True),
        ("sí, termina el quiz", True),
        ("no, para el quiz porfa", True),
        ("sí, para el examen porfa", False),
        ("sí, para el quiz por favor", False),
        ("falta una y ya termina el quiz?", False),
        ("me falta una y ya termina el examen verdad?", False),
        ("¿y si fallo, termina el quiz?", False),
        ("y si fallo, termina el quiz", False),
        ("ya me cansé, termina el quiz?", False),
        # by design: a plain statement is read as a stop; the server confirms before ending the quiz
        ("falta una y ya termina el quiz", True),
        # "no le sigo" may be "I don't follow you"
        ("no le sigo", False),
        # by design: it may be asking how to finish
        ("¿cómo le hago para terminar el quiz?", False),
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
        ("¿cómo se factoriza un trinomio?", "factorizar un trinomio"),
        ("como se factoriza un binomio", "factorizar un binomio"),
        ("¿cómo se simplifican los racionales?", "simplificar los racionales"),
        ("¿cómo se le saca la raíz a un número?", "sacar la raíz a un número"),
        ("como le saco la raiz a un numero", "sacar la raiz a un numero"),
        ("¿por qué no me enseñas fracciones?", "fracciones"),
        # what follows "y luego" is the next request, not the topic
        ("explícame fracciones y luego me haces un quiz", "fracciones"),
        ("explícame fracciones y también hazme un quiz", "fracciones"),
        ("quiero aprender fracciones y también decimales", "fracciones y también decimales"),
        ("explícame fracciones y después un quiz", "fracciones"),
        ("explícame fracciones y luego decimales", "fracciones y luego decimales"),
        ("enséñame el antes y después de la contaminación", "el antes y después de la contaminación"),
        # capped at MAX_TOPIC_WORDS, without the dangling "de"
        ("enséñame el antes y después de la contaminación de la laguna", "el antes y después de la contaminación"),
        # "¿qué es ...?" about a math idea asks to have it explained: a lesson with a worked example
        ("¿Qué es una fracción?", "una fracción"),
        ("que es una fraccion", "una fraccion"),
        ("¿Qué son las fracciones equivalentes?", "las fracciones equivalentes"),
        ("¿qué significa denominador?", "denominador"),
        ("¿qué quiere decir ecuación?", "ecuación"),
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
        # no topic to teach
        "quiero aprender más",
        "quiero aprender mas",
        "quiero aprender lo de ayer",
        "quiero aprender todo",
        "quiero aprender más cosas",
        "quiero aprender lo de la clase",
        "quiero aprender lo que vimos en clase",
        "quiero aprender el tema de hoy",
        # "¿qué es ...?" without a math idea, or about the board
        "¿qué es esto?",
        "¿qué es tres más cinco?",
        "¿cuál es la respuesta?",
        "¿qué es lo que sigue?",
        "¿qué significa eso?",
    ],
)
def test_detect_learn_request_es_ignores(text: str) -> None:
    assert detect_learn_request_es(text) is None


def test_learn_topic_is_capped() -> None:
    topic = detect_learn_request_es(
        "quiero aprender a resolver ecuaciones de primer grado con fracciones y decimales juntos"
    )
    assert topic == "resolver ecuaciones de primer grado con fracciones"
    assert len(topic.split()) <= lang.MAX_TOPIC_WORDS


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
        ("tengo uno pero está difícil", "own"),
        ("tengo uno pero esta dificil", "own"),
        ("simon, tengo uno", "own"),
        ("yo lo hago", "own"),
        ("lo hago yo", "own"),
        ("tengo los míos", "own"),
        ("tengo los mios", "own"),
        ("tengo mis propios ejercicios", "own"),
        ("cualquiera", "generate"),
        ("dame cualquiera", "generate"),
        ("el que tú quieras", "generate"),
        ("el que tu quieras", "generate"),
        ("como tu digas", "generate"),
        ("lo que sea", "generate"),
        ("mándame uno", "generate"),
        ("echame uno", "generate"),
        ("aviéntame uno", "generate"),
        ("uno tuyo", "generate"),
        ("mejor tú", "generate"),
        ("mejor tu", "generate"),
        ("me da igual", "generate"),
        ("dame otro ejercicio", "generate"),
        ("no, ninguno", "generate"),
        ("no, nada", "generate"),
        ("no, mejor tú", "generate"),
        ("uno fácil porfa", "generate"),
        ("sí, uno fácil", "generate"),
        ("uno de fracciones", "generate"),
        ("no lo tengo aquí", "generate"),
        ("mejor yo", "own"),
        ("yo te doy uno", "own"),
        ("yo te digo uno", "own"),
        ("ya lo escribí", "own"),
        ("ya lo escribí en el pizarrón", "own"),
        ("ahí te va", "own"),
        ("sí traigo", "own"),
        ("uno de la tarea", "own"),
        ("sí, de mi libro", "own"),
        ("tengo uno bien difícil", "own"),
        ("no, yo lo hago", "own"),
        ("¿dónde lo escribo?", "own"),
        ("no lo tengo aquí, pero te lo dicto", "own"),
        ("te lo dicto", "own"),
        ("te lo digo", "own"),
        ("uno del libro", "own"),
        ("el de la página 45", "own"),
        # Whisper drops the comma of "no, yo lo hago"
        ("no yo lo hago", "own"),
        ("no yo lo escribo", "own"),
        ("tengo uno bien largo", "own"),
        ("tengo uno bien padre", "own"),
        ("tengo uno bueno", "own"),
        ("uno de porcentajes con descuentos", "generate"),
        ("uno con decimales", "generate"),
        ("uno de 3 pasos", "generate"),
        ("hazme uno con decimales", "generate"),
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
        "tengo una muy buena idea",
        "tengo una bien buena pregunta",
        "tengo una bien rara duda",
        "tengo una facil pregunta",
        "no lo hago yo",
        "yo lo hago mal",
        "no lo escribo",
        "una de mis dudas",
        "no lo tengo claro",
        "yo te digo una cosa",
        "uno de tarea",
        "no te lo digo",
        "tengo una buena noticia",
        "tengo hambre",
        "no entiendo",
        "make one up",
        "",
    ],
)
def test_detect_offer_reply_es_ignores(text: str) -> None:
    assert detect_offer_reply_es(text) is None


# ---- long input ------------------------------------------------------------------
def test_input_past_the_cap_is_ignored() -> None:
    padding = "x " * lang.MAX_INPUT_CHARS
    assert detect_mode_es("examíname " + padding) == "quiz"
    assert detect_mode_es(padding + "examíname") is None
    assert is_quiz_stop_es(padding + "termina el quiz") is False
    assert detect_learn_request_es(padding + "quiero aprender fracciones") is None
    assert detect_offer_reply_es(padding + "invéntame uno") is None
    assert detect_language("2 " * lang.MAX_INPUT_CHARS + "hola, ¿qué sigue?", "en") == "en"


# The limit is far above the few milliseconds these take: it catches a runaway pattern, not a slow runner.
def _seconds(call: Callable[..., object], *args: object, **kwargs: object) -> float:
    start = time.perf_counter()
    call(*args, **kwargs)
    return time.perf_counter() - start


def test_pathological_input_stays_fast(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", None)
    assert _seconds(detect_all, "no quiero un quiz " * 3000) < 2.0
    assert _seconds(detect_learn_request_es, "aprender " + "porfa " * 10000) < 2.0
    assert _seconds(detect_language, "y " * 50000) < 2.0


# A fast machine runs quadratic code on these inputs under any generous time limit, so the internals are checked
# by growth instead: 4x the input takes about 4x as long when linear and about 16x when quadratic.
_LINEAR_GROWTH = 8.0


def _best_of_three(call: Callable[[], object]) -> float:
    return min(_seconds(call) for _ in range(3))


def _growth(make: Callable[[int], Callable[[], object]], size: int, factor: int = 4) -> float:
    small = _best_of_three(make(size))
    assert small < 1.0, "already far too slow; not timing the larger input"
    return _best_of_three(make(size * factor)) / small


def _stop_unasked(text: str) -> bool:
    return lang._opens_with_stop(text, asked=False)


# Each input is rejected at every match, so every lookback runs.
@pytest.mark.parametrize(
    ("check", "unit", "size"),
    [
        pytest.param(lambda text: lang._commanded(lang._QUIZ_ES, text), "no quiero un quiz ", 1000, id="commanded"),
        pytest.param(lambda text: lang._requested(lang._QUIZ_ES, text), "no quiero un quiz ", 1000, id="requested"),
        pytest.param(
            lambda text: lang._requested(lang._QUIZ_ES, text), "despues me haces un quiz , ", 1000, id="put-off"
        ),
        pytest.param(
            lambda text: lang._requested(
                lang._PRACTICE_ES, text, unless=lambda plain, found: lang._describes(plain, found, asked=True)
            ),
            "hay otro problema asi en la hoja ",
            500,
            id="statement",
        ),
        pytest.param(
            lambda text: lang._requested(lang._QUIZ_ES, text),
            "pero manana en la tarde hazme un quiz ",
            1000,
            id="later",
        ),
        pytest.param(_stop_unasked, "ya me canse , termina el quiz a las tres , ", 1000, id="clauses"),
        pytest.param(_stop_unasked, "ya me canse pero para el examen de manana ", 1000, id="conjunctions"),
        pytest.param(_stop_unasked, "ya me canse termina el quiz a las tres , ", 1000, id="reason-first"),
        pytest.param(_stop_unasked, "ya ya me canse ya ya termina el quiz a las tres , ", 1000, id="reason-fillers"),
        pytest.param(lang._stuck, "no estoy atorado ", 2000, id="stuck"),
        pytest.param(lang._offered, "no lo hago yo ", 2000, id="offered"),
    ],
)
def test_lookback_is_linear(check: Callable[[str], bool], unit: str, size: int) -> None:
    assert not check(unit * size)
    assert _growth(lambda count: lambda: check(unit * count), size) < _LINEAR_GROWTH


def test_filler_trim_is_linear() -> None:
    def trim(count: int) -> Callable[[], object]:
        words = ["porfa"] * count
        return lambda: lang._trim(words, words, lang._TRAILING_FILLER_ES, leading=False)

    assert _growth(trim, 2000) < _LINEAR_GROWTH


def test_variable_check_does_not_copy_the_text() -> None:
    def checks(size: int) -> Callable[[], object]:
        text = "y " * size
        middle = len(text) // 2

        def check() -> None:
            for start in range(middle, middle + 4000, 2):
                lang._is_variable(text, start, start + 1)

        return check

    # The same 2000 checks on a text 8x as long: as fast if nothing is copied, about 8x slower if it is.
    assert _growth(checks, 250_000, factor=8) < 3.0


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
    assert detect_all("explícame fracciones y luego me haces un quiz") == {
        "language": "es",
        "mode": "quiz",
        "learn_topic": "fracciones",
        "offer_reply": None,
    }


def test_detect_all_without_tutor_module(
    fresh_english: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", None)
    with caplog.at_level(logging.WARNING, logger="biobench_sar.lang"):
        assert lang._english_detectors() is None
        assert detect_all("quiz me") == {"language": "en", "mode": None, "learn_topic": None, "offer_reply": None}
    assert not caplog.records


class _BrokenTutor(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds biobench_sar.tutor, then fails importing it the way a broken dependency would."""

    def __init__(self, error: ImportError) -> None:
        self.error = error

    def find_spec(self, name: str, path: object = None, target: object = None) -> importlib.machinery.ModuleSpec | None:
        return importlib.util.spec_from_loader(name, self) if name == "biobench_sar.tutor" else None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        ImportError("cannot import name 'QuizBank' from 'biobench_sar.rag.quiz_bank'"),
        ModuleNotFoundError("No module named 'chromadb'", name="chromadb"),
    ],
)
def test_a_broken_tutor_is_logged(
    error: ImportError, fresh_english: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "biobench_sar.tutor", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenTutor(error), *sys.meta_path])
    with caplog.at_level(logging.WARNING, logger="biobench_sar.lang"):
        assert lang._english_detectors() is None
    assert any("biobench_sar.tutor" in record.getMessage() for record in caplog.records)
    assert detect_all("examíname")["mode"] == "quiz"


def test_a_half_imported_tutor_is_looked_at_again(fresh_english: None, monkeypatch: pytest.MonkeyPatch) -> None:
    loading = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "biobench_sar.tutor", loading)
    assert lang._english_detectors() is None
    finished = _fake_tutor("quiz")
    for name in ("detect_mode", "detect_learn_request", "detect_offer_reply"):
        setattr(loading, name, getattr(finished, name))
    assert detect_all("quiz me")["mode"] == "quiz"


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
