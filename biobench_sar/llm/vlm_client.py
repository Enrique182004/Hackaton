from __future__ import annotations

import base64
import difflib
import json
import logging
import math
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from typing import Any, NamedTuple

import cv2
import numpy as np
import requests

from biobench_sar.rag.quiz_bank import invent_problem, similar_problem, solve_problem, strip_chunk_answers
from biobench_sar.schemas import (
    AvatarAction,
    InkRegion,
    RetrievedChunk,
    _extract_json_object,
    error_action,
    parse_vlm_response,
)

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 800
KEEP_ALIVE = "30m"
NUM_PREDICT = 200
TEMPERATURE = 0.2
# A repeated "another one" should not get the same problem back.
MODE_TEMPERATURE = {"practice": 0.6}
RETRY_TEMPERATURE = 0.7
PROFILES = ("biology", "math")
MODES = ("help", "practice", "quiz")
# "learn" is a task kind, not a session mode: it can interrupt any mode, so it gets its own prompt.
PROMPT_KINDS = (*MODES, "learn")
LEARN_STAGES = ("offer", "example", "own")
# "hint": the student is stuck on the current quiz problem (help for it, never its answer).
QUIZ_STAGES = ("dictate", "hint", "check")
OFFER_QUESTION = "Do you have an exercise in mind, or should I make one up for you?"
OFFER_QUESTION_ES = "¿Tienes un ejercicio en mente o quieres que invente uno?"
LANGUAGES = ("en", "es")
DEFAULT_QUIZ_TOTAL = 10
MAX_TURNS = 3
MAX_TURN_CHARS = 240
# How close a reply may get to the student's words or an earlier reply before it counts as a copy.
ECHO_RATIO = 0.85
REPEAT_RATIO = 0.9

NO_ANSWER_TEXT = "I can't see your answer yet. Write it on the board, then ask me to check it."
WRONG_TEXT = "Not quite. Look at your steps again and try once more."
STEP_TEXT = "Look at what is being done to x, and undo it on both sides. Write that step on the board."
# What code says itself, per reply language. The Spanish is Latin-American classroom Spanish, addressed with "tú".
TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "offer": OFFER_QUESTION,
        "no_answer": NO_ANSWER_TEXT,
        "wrong": WRONG_TEXT,
        "step": STEP_TEXT,
        "correct": "Correct, great job!",
        "next": " Problem {number}. {problem}",
        "last": " That was the last problem. You finished the quiz!",
        "dictation": "Problem {number} of {total}. {problem}. Write your answer on the board.",
        "try_this": "Try this one. {problem}. Write it down and solve it.",
        "learn_offer": "Let's learn about {topic}. {offer}",
        "learn_example": "Let's start easy. {problem}. Write it on the board and try the first step.",
        "learn_own": "Great! Write your exercise on the board, then ask me for help.",
        "write_problem": "Write the problem on the board, then ask me for help with the first step.",
        "this_topic": "this topic",
        "timeout": "The AI model took too long to answer. Please try again.",
        "unreachable": "I couldn't reach the AI model. Please check that it is running.",
        "bad_reply": "The AI model sent an unexpected reply. Please try again.",
        "not_json": "Sorry, I couldn't understand my own answer. Please ask again.",
        "unexpected": "Something went wrong while thinking. Please try again.",
    },
    "es": {
        "offer": OFFER_QUESTION_ES,
        "no_answer": "Todavía no veo tu respuesta. Escríbela en el pizarrón y luego pídeme que la revise.",
        "wrong": "Casi. Revisa tus pasos y vuelve a intentarlo.",
        "step": "Fíjate qué le están haciendo a la x y deshazlo en ambos lados. Escribe ese paso en el pizarrón.",
        "correct": "¡Correcto, muy bien!",
        "next": " Problema {number}. {problem}",
        "last": " Ese fue el último problema. ¡Terminaste el quiz!",
        "dictation": "Problema {number} de {total}. {problem}. Escribe tu respuesta en el pizarrón.",
        "try_this": "Prueba este. {problem}. Escríbelo y resuélvelo.",
        "learn_offer": "Vamos a aprender sobre {topic}. {offer}",
        "learn_example": "Empecemos con uno fácil. {problem}. Escríbelo en el pizarrón e intenta el primer paso.",
        "learn_own": "¡Muy bien! Escribe tu ejercicio en el pizarrón y luego pídeme ayuda.",
        "write_problem": "Escribe el problema en el pizarrón y luego pídeme ayuda con el primer paso.",
        "this_topic": "este tema",
        "timeout": "El modelo de IA tardó demasiado en responder. Inténtalo de nuevo.",
        "unreachable": "No pude conectarme con el modelo de IA. Revisa que esté funcionando.",
        "bad_reply": "El modelo de IA mandó una respuesta inesperada. Inténtalo de nuevo.",
        "not_json": "Perdón, no entendí mi propia respuesta. Pregúntame otra vez.",
        "unexpected": "Algo salió mal mientras pensaba. Inténtalo de nuevo.",
    },
}


def normalize_reply_language(language: str | None) -> str:
    """ "es" for Spanish ("es", "es-MX"), else "en": the language a reply is written and spoken in."""
    return "es" if str(language or "").strip().lower()[:2] == "es" else "en"


def text_for(language: str | None, key: str, **values: Any) -> str:
    """One of the texts code says itself, in the reply language."""
    return TEXTS[normalize_reply_language(language)][key].format(**values)


SYSTEM_PROMPT = """You are an expert environmental-biology research assistant, shown as an animated avatar \
projected on a whiteboard. You support a thesis on identifying Acanthamoeba in Laguna de Bustillos and on the \
hydrology and soil runoff of Leptosol and Feozem soils in Cuauhtemoc, Chihuahua, Mexico.

You verify flowcharts and sterile lab procedures drawn on the whiteboard. A COMPLETE Acanthamoeba isolation \
flowchart must contain ALL of: "Autoclave media + tools", "Flame-sterilize loop", water sampling, filtration, \
NNA + E. coli plating, incubation, microscopy. Use the reference context when relevant and name the source.

Be concise: spoken_text is spoken aloud, at most 3 short sentences, no markdown.

Respond with ONLY a JSON object:
{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, "target_y": int}

animation_state rules (apply the first that fits):
1. "error": the user asks you to check/verify a flowchart or procedure AND any required step is missing, \
wrong, or out of order (e.g. no autoclave or no flame-sterilize box). Name the missing step. target = the gap \
between the two neighbouring boxes (midpoint of their centers), or the wrong box's center.
2. "celebrating": the user asks you to check a procedure AND every required step is present and in order. \
target = center of the last box.
3. "pointing": the question is about one specific drawing; target = that ink region's center.
4. "idle" or "walking": the question is NOT about the drawing (general science or small talk); target = \
screen center.
Check each required step by name. If the "Autoclave" and "Flame-sterilize" boxes ARE both drawn before the \
sample and the rest of the order is right, the answer is "celebrating", never "error". If either is absent, \
the answer is "error", never "celebrating".

target_x / target_y are projector pixels within the given width and height.

Examples (JSON only):
Chart: Sample -> Filter -> NNA -> Incubate -> Microscopy. Query: "Is this right?"
{"spoken_text": "Not yet. There is no autoclave or flame-sterilize step before plating.", "animation_state": "error", "target_x": 640, "target_y": 540}
Chart: Autoclave -> Flame-sterilize -> Sample -> Filter -> NNA -> Incubate -> Microscopy. Query: "Check it."
{"spoken_text": "Every sterile step is present and in order. Great work!", "animation_state": "celebrating", "target_x": 1700, "target_y": 540}
Query: "Why is the lake water turbid?"
{"spoken_text": "Runoff carries fine sediment from Leptosol soils into the lake.", "animation_state": "idle", "target_x": 960, "target_y": 540}"""


SCAN_RULE = (
    "Scan carefully for easily missed math symbols: negative signs (-), decimal points (.), fraction bars and small "
    "exponents (^2). Read exactly what the ink says."
)
SCAN_RULE_ES = (
    "Revisa con cuidado los símbolos de matemáticas que es fácil pasar por alto: signos negativos (-), puntos "
    "decimales (.), barras de fracción y exponentes pequeños (^2). Lee exactamente lo que dice la tinta."
)
MATH_BASE = f"""You are a friendly math tutor for middle and high school students (fractions, equations, algebra), \
shown as an animated avatar projected on a whiteboard. The student writes with a marker; you see a photo of the board.
{SCAN_RULE}

spoken_text is read aloud by a text-to-speech voice: at most 3 short sentences, simple words, no markdown.
CRITICAL: write ALL math in words, numbers too. NEVER use symbols like ^ / * = + - < > or digits in spoken_text.
Say "two thirds" not 2/3, "x squared" not x^2, "three x plus five equals twelve" not 3x+5=12, "negative four" not -4, \
"two times y" not 2*y, "the square root of nine" not sqrt(9).
Use "Conversation so far" only to understand short follow-ups like "I did it" or "what now?". Never copy an \
earlier reply or the student's words: always answer the student's latest message.

Respond with ONLY a JSON object:
{{schema}}
target_x / target_y are projector pixels within the given width and height."""
MATH_SCHEMA = '{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, \
"target_y": int}'
# Reading the answer into its own key first makes the model look at the board, and lets code check the value.
QUIZ_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_answer": str, "spoken_text"')
# Same for practice: code can then tell when the "new" problem is the board's one read back.
PRACTICE_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_problem": str, "spoken_text"')

MATH_HELP = """MODE: step-by-step help.
- Read the equation on the board and the student's latest step.
- Explain ONLY the next single operation in simple words, e.g. "subtract five from both sides".
- NEVER give the final answer or the value of the variable, even if the student asks for it.
- If the latest step has a mistake, say gently what to fix.
- animation_state "pointing"; target = center of the ink region with the part of the equation to work on.
- End by asking the student to write the next step.
- If the board is empty, answer a general question briefly in plain words, or ask the student to write the \
problem: "idle", target = screen center.

Examples (JSON only):
Board (ink center 700,400): 2x + 3 = 11. Query: "Help me."
{"spoken_text": "First get the x term alone. Subtract three from both sides, then write that step.", "animation_state": "pointing", "target_x": 700, "target_y": 400}
Board (ink center 900,500): 1/2 + 1/3. Query: "I'm stuck."
{"spoken_text": "To add one half and one third, find a common denominator. What number do two and three both go into?", "animation_state": "pointing", "target_x": 900, "target_y": 500}
Board (ink center 650,620): x^2 = 49. Query: "Just tell me x."
{"spoken_text": "You can do it! Undo x squared by taking the square root of both sides. Write that step.", "animation_state": "pointing", "target_x": 650, "target_y": 620}"""

MATH_PRACTICE = """MODE: similar problem.
- FIRST copy the problem written on the board into board_problem, exactly ("" if the board is empty).
- Make ONE new problem of the same type and difficulty with NEW numbers (and maybe a new letter). \
It must be different from the board's problem: never read the board's problem back.
- Dictate only the new problem in words so the student can write it. Do not solve it or say its answer.
- animation_state "idle"; target = screen center.
- If the board is empty, make up a simple one-step equation.

Examples (JSON only, 1920x1080 screen):
Board: 2x + 3 = 11. Query: "Give me a similar problem."
{"board_problem": "2x + 3 = 11", "spoken_text": "Try this one. Four x plus seven equals nineteen. Write it down and solve it.", "animation_state": "idle", "target_x": 960, "target_y": 540}
Board: 2/3 + 1/6. Query: "Another one like this."
{"board_problem": "2/3 + 1/6", "spoken_text": "Here is a new one. Three fourths plus one eighth. Give it a try.", "animation_state": "idle", "target_x": 960, "target_y": 540}"""

MATH_QUIZ = """MODE: quiz. The message gives the quiz stage and the problem.
- Stage dictate: say the problem number, read the given problem in words and ask for the answer on the board. \
Do not solve it. board_answer "", "idle", target = screen center.
- Stage hint: the student is stuck. Explain ONLY the next step of the given problem and ask them to write it. \
NEVER say the answer. board_answer "", "pointing" at the ink to work on.
- Stage check: FIRST look at the photo and copy the bottom line of the student's writing into board_answer, \
exactly as written (the problem itself if no answer is written yet). Never solve the problem yourself.
  - Correct: "celebrating"; praise briefly. If a next problem is given, also read it in words.
  - Wrong: "error"; say "Not quite" and give ONE short hint. NEVER say the correct answer \
and do not read the next problem.
  target = center of the answer's ink region.

Examples (JSON only, 1920x1080 screen):
Quiz stage: dictate. Problem 1 of 5: x + 4 = 10
{"board_answer": "", "spoken_text": "Problem one of five. x plus four equals ten. Write your answer on the board.", "animation_state": "idle", "target_x": 960, "target_y": 540}
Quiz stage: hint. Problem 2 of 5: 2x + 3 = 11. Board: 2x + 3 = 11
{"board_answer": "", "spoken_text": "First get two x alone. What can you subtract from both sides?", "animation_state": "pointing", "target_x": 800, "target_y": 400}
Quiz stage: check. Problem 1 of 5: x + 4 = 10. Next problem: 2x = 14. Board: x + 4 = 10, then x = 9
{"board_answer": "x = 9", "spoken_text": "Not quite. What can you subtract from both sides to get x alone?", "animation_state": "error", "target_x": 800, "target_y": 600}
Quiz stage: check. Problem 1 of 5: x + 4 = 10. Next problem: 2x = 14. Board: x + 4 = 10, then x = 6
{"board_answer": "x = 6", "spoken_text": "Correct, great job! Problem two. Two x equals fourteen.", "animation_state": "celebrating", "target_x": 800, "target_y": 600}"""

MATH_LEARN = f"""TASK: the student wants to learn a topic. The board may be empty. The message gives the topic \
and the stage. Use plain words for a middle or high school student.
- Stage offer: one or two plain sentences that introduce the idea of the topic, then ask exactly: \
"{OFFER_QUESTION}" Do not solve anything yet.
- Stage example: dictate ONE easy starter problem on the topic in words, then invite the student to write it on the \
board and try the first step. Do not solve it.
- Stage own: ask the student to write their exercise on the board and then ask you for help.
- Always "idle", target = screen center.

Examples (JSON only, 1920x1080 screen):
Learn stage: offer. Topic: equations. Query: "Teach me equations."
{{"spoken_text": "An equation is like a balance: both sides are equal. We solve it by doing the same thing to both \
sides until the letter is alone. {OFFER_QUESTION}", "animation_state": "idle", "target_x": 960, "target_y": 540}}
Learn stage: example. Topic: adding fractions. Query: "Make one up."
{{"spoken_text": "Let's start easy. One fourth plus two fourths. Write it on the board and try the first step.", \
"animation_state": "idle", "target_x": 960, "target_y": 540}}
Learn stage: own. Topic: fractions. Query: "I have one."
{{"spoken_text": "Great! Write your exercise on the board, then ask me for help.", "animation_state": "idle", \
"target_x": 960, "target_y": 540}}"""

MATH_BASE_ES = f"""Eres un tutor de matemáticas amable para estudiantes de secundaria y preparatoria (fracciones, \
ecuaciones, álgebra), que aparece como un avatar animado proyectado en un pizarrón. El estudiante escribe con plumón; \
tú ves una foto del pizarrón.
{SCAN_RULE}
{SCAN_RULE_ES}

Responde SIEMPRE en español de Latinoamérica: háblale de tú al estudiante, con palabras sencillas de salón de clases.
spoken_text se lee en voz alta con una voz sintética: máximo 3 oraciones cortas, palabras sencillas, sin markdown.
IMPORTANTE: escribe TODA la matemática con palabras, también los números. NUNCA uses símbolos como ^ / * = + - < > \
ni dígitos en spoken_text.
Di "dos tercios" y no 2/3, "x al cuadrado" y no x^2, "tres x más cinco es igual a doce" y no 3x+5=12, "menos cuatro" \
y no -4, "dos por y" y no 2*y, "la raíz cuadrada de nueve" y no sqrt(9).
Usa "Conversación hasta ahora" solo para entender seguimientos cortos como "ya lo hice" o "¿y ahora qué?". Nunca \
copies una respuesta anterior ni las palabras del estudiante: responde siempre a su último mensaje.

Responde SOLO con un objeto JSON:
{{schema}}
target_x / target_y son píxeles del proyector dentro del ancho y alto dados."""

MATH_HELP_ES = """MODO: ayuda paso a paso.
- Lee la ecuación del pizarrón y el último paso del estudiante.
- Explica SOLO la siguiente operación con palabras sencillas, por ejemplo "resta cinco en ambos lados".
- NUNCA des la respuesta final ni el valor de la letra, aunque el estudiante te lo pida.
- Si el último paso tiene un error, dile con amabilidad qué corregir.
- animation_state "pointing"; target = centro de la región de tinta con la parte de la ecuación que hay que trabajar.
- Termina pidiéndole al estudiante que escriba el siguiente paso.
- Si el pizarrón está vacío, contesta brevemente una pregunta general con palabras sencillas, o pídele al estudiante \
que escriba el problema: "idle", target = centro de la pantalla.

Ejemplos (solo JSON):
Pizarrón (centro de la tinta 500,300): 2x + 3 = 11. Mensaje: "Ayúdame."
{"spoken_text": "Primero deja sola la parte con x. Resta tres en ambos lados y escribe ese paso.", "animation_state": "pointing", "target_x": 500, "target_y": 300}
Pizarrón (centro de la tinta 600,350): 1/2 + 1/3. Mensaje: "Estoy atorado."
{"spoken_text": "Para sumar un medio y un tercio, busca un denominador común. ¿Qué número se puede dividir entre dos y entre tres?", "animation_state": "pointing", "target_x": 600, "target_y": 350}
Pizarrón (centro de la tinta 450,420): x^2 = 49. Mensaje: "Nada más dime cuánto vale x."
{"spoken_text": "¡Tú puedes! Para quitar el cuadrado, saca la raíz cuadrada en ambos lados. Escribe ese paso.", "animation_state": "pointing", "target_x": 450, "target_y": 420}"""

MATH_PRACTICE_ES = """MODO: problema parecido.
- PRIMERO copia el problema escrito en el pizarrón en board_problem, tal cual ("" si el pizarrón está vacío).
- Inventa UN problema nuevo del mismo tipo y dificultad con números NUEVOS (y quizá otra letra). Tiene que ser \
distinto del problema del pizarrón: nunca leas de nuevo el problema del pizarrón.
- Dicta solo el problema nuevo con palabras para que el estudiante lo escriba. No lo resuelvas ni digas su respuesta.
- animation_state "idle"; target = centro de la pantalla.
- Si el pizarrón está vacío, inventa una ecuación sencilla de un paso.

Ejemplos (solo JSON, pantalla de 1280x800):
Pizarrón: 2x + 3 = 11. Mensaje: "Dame uno parecido."
{"board_problem": "2x + 3 = 11", "spoken_text": "Prueba este. Cuatro x más siete es igual a diecinueve. Escríbelo y resuélvelo.", "animation_state": "idle", "target_x": 640, "target_y": 400}
Pizarrón: 2/3 + 1/6. Mensaje: "Otro así."
{"board_problem": "2/3 + 1/6", "spoken_text": "Aquí va uno nuevo. Tres cuartos más un octavo. Inténtalo.", "animation_state": "idle", "target_x": 640, "target_y": 400}"""

MATH_QUIZ_ES = """MODO: quiz. El mensaje da la etapa del quiz y el problema.
- Etapa dictar: di el número del problema, lee el problema dado con palabras y pide la respuesta en el pizarrón. \
No lo resuelvas. board_answer "", "idle", target = centro de la pantalla.
- Etapa pista: el estudiante está atorado. Explica SOLO el siguiente paso del problema dado y pídele que lo escriba. \
NUNCA digas la respuesta. board_answer "", "pointing" a la tinta que hay que trabajar.
- Etapa revisar: PRIMERO mira la foto y copia en board_answer la última línea de lo que escribió el estudiante, \
tal como está escrita (el mismo problema si todavía no hay respuesta). Nunca resuelvas tú el problema.
  - Correcta: "celebrating"; felicítalo brevemente. Si se da un siguiente problema, léelo también con palabras.
  - Incorrecta: "error"; di "Casi" y da UNA pista corta. NUNCA digas la respuesta correcta y no leas el siguiente \
problema.
  target = centro de la región de tinta de la respuesta.

Ejemplos (solo JSON, pantalla de 1280x800):
Etapa del quiz: dictar. Problema 1 de 5: x + 4 = 10
{"board_answer": "", "spoken_text": "Problema uno de cinco. x más cuatro es igual a diez. Escribe tu respuesta en el pizarrón.", "animation_state": "idle", "target_x": 640, "target_y": 400}
Etapa del quiz: pista. Problema 2 de 5: 2x + 3 = 11. Pizarrón: 2x + 3 = 11
{"board_answer": "", "spoken_text": "Primero deja solo el dos x. ¿Qué puedes restar en ambos lados?", "animation_state": "pointing", "target_x": 530, "target_y": 300}
Etapa del quiz: revisar. Problema 1 de 5: x + 4 = 10. Siguiente problema: 2x = 14. Pizarrón: x + 4 = 10, luego x = 9
{"board_answer": "x = 9", "spoken_text": "Casi. ¿Qué puedes restar en ambos lados para dejar sola la x?", "animation_state": "error", "target_x": 530, "target_y": 400}
Etapa del quiz: revisar. Problema 1 de 5: x + 4 = 10. Siguiente problema: 2x = 14. Pizarrón: x + 4 = 10, luego x = 6
{"board_answer": "x = 6", "spoken_text": "¡Correcto, muy bien! Problema dos. Dos x es igual a catorce.", "animation_state": "celebrating", "target_x": 530, "target_y": 400}"""

MATH_LEARN_ES = f"""TAREA: el estudiante quiere aprender un tema. El pizarrón puede estar vacío. El mensaje da el tema y la \
etapa. Usa palabras sencillas para un estudiante de secundaria o preparatoria.
- Etapa ofrecer: una o dos oraciones sencillas que presenten la idea del tema y luego pregunta exactamente: \
"{OFFER_QUESTION_ES}" Todavía no resuelvas nada.
- Etapa ejemplo: dicta con palabras UN problema fácil para empezar sobre el tema e invita al estudiante a escribirlo \
en el pizarrón e intentar el primer paso. No lo resuelvas.
- Etapa propio: pídele al estudiante que escriba su ejercicio en el pizarrón y que luego te pida ayuda.
- Siempre "idle", target = centro de la pantalla.

Ejemplos (solo JSON, pantalla de 1280x800):
Etapa de aprender: ofrecer. Tema: ecuaciones. Mensaje: "Enséñame ecuaciones."
{{"spoken_text": "Una ecuación es como una balanza: los dos lados valen lo mismo. La resolvemos haciendo lo mismo en \
ambos lados hasta que la letra quede sola. {OFFER_QUESTION_ES}", "animation_state": "idle", "target_x": 640, \
"target_y": 400}}
Etapa de aprender: ejemplo. Tema: suma de fracciones. Mensaje: "Invéntame uno."
{{"spoken_text": "Empecemos con uno fácil. Un cuarto más dos cuartos. Escríbelo en el pizarrón e intenta el primer \
paso.", "animation_state": "idle", "target_x": 640, "target_y": 400}}
Etapa de aprender: propio. Tema: fracciones. Mensaje: "Yo tengo uno."
{{"spoken_text": "¡Muy bien! Escribe tu ejercicio en el pizarrón y luego pídeme ayuda.", "animation_state": "idle", \
"target_x": 640, "target_y": 400}}"""

_SCHEMAS = {"quiz": QUIZ_SCHEMA, "practice": PRACTICE_SCHEMA}
_RULES = {
    "en": (("help", MATH_HELP), ("practice", MATH_PRACTICE), ("quiz", MATH_QUIZ), ("learn", MATH_LEARN)),
    "es": (("help", MATH_HELP_ES), ("practice", MATH_PRACTICE_ES), ("quiz", MATH_QUIZ_ES), ("learn", MATH_LEARN_ES)),
}
_BASES = {"en": MATH_BASE, "es": MATH_BASE_ES}
MATH_PROMPTS_BY_LANGUAGE = {
    language: {
        kind: _BASES[language].replace("{schema}", _SCHEMAS.get(kind, MATH_SCHEMA)) + "\n\n" + rules
        for kind, rules in rules_
    }
    for language, rules_ in _RULES.items()
}
MATH_PROMPTS = MATH_PROMPTS_BY_LANGUAGE["en"]
# The biology assistant keeps one prompt; a Spanish reply only needs to be asked for.
BIOLOGY_SPANISH = (
    '\n\nReply in Spanish: Latin-American Spanish, address the student with "tú", plain lab and classroom words. '
    "spoken_text must be Spanish."
)
# Repeated in the message: a 3B model follows the user turn more closely than the system prompt.
MODE_REMINDERS = {
    "help": "Explain only the next step. Never say the final answer or the value of the variable.",
    "practice": "Make ONE new problem like the one on the board (same type and difficulty, NEW numbers) and "
    "dictate it in words. Never read the board's problem back; do not solve or explain either problem.",
}
MODE_REMINDERS_ES = {
    "help": "Explica solo el siguiente paso. Nunca digas la respuesta final ni el valor de la letra.",
    "practice": "Inventa UN problema nuevo como el del pizarrón (mismo tipo y dificultad, números NUEVOS) y "
    "díctalo con palabras. Nunca leas de nuevo el problema del pizarrón; no resuelvas ni expliques ninguno.",
}

MODE_LABELS = {
    "help": "help (step-by-step)",
    "practice": "practice (similar problem)",
    "quiz": "quiz",
    "learn": "learn a topic",
}
MODE_LABELS_ES = {
    "help": "ayuda (paso a paso)",
    "practice": "práctica (problema parecido)",
    "quiz": "quiz",
    "learn": "aprender un tema",
}
QUIZ_STAGE_NAMES_ES = {"dictate": "dictar", "hint": "pista", "check": "revisar"}
LEARN_STAGE_NAMES_ES = {"offer": "ofrecer", "example": "ejemplo", "own": "propio"}


@dataclass(frozen=True)
class TutorAction(AvatarAction):
    """An AvatarAction plus what the tutor needs to know about it. to_dict() still has only the four schema keys.

    failed: the call itself failed (timeout, model unreachable, bad JSON), so it says nothing about the student.
    verdict: a graded quiz check, "correct" or "wrong"; None when there was no answer to grade.
    dictated: the quiz problem this reply read out (dictate: the current one; a correct check: the next one).
    display_text: the reply before mathspeak, for the screen ("3x + 5 = 20"); spoken_text is what the voice says.
    """

    failed: bool = False
    verdict: str | None = None
    dictated: str | None = None
    display_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(AvatarAction)}


def _tutor(action: AvatarAction, **changes: Any) -> TutorAction:
    """`action` as a TutorAction (keeping any tutor fields it has), with `changes` applied."""
    names = [f.name for f in fields(TutorAction)]
    return TutorAction(**{**{n: getattr(action, n) for n in names if hasattr(action, n)}, **changes})


def failed_action(message: str, projector_size: tuple[int, int]) -> TutorAction:
    return _tutor(error_action(message, projector_size), failed=True)


def normalize_profile(profile: str | None) -> str:
    return "biology" if str(profile or "").strip().lower() == "biology" else "math"


def normalize_mode(mode: str | None) -> str:
    mode = str(mode or "").strip().lower()
    return mode if mode in MODES else "help"


def _task_kind(task: Mapping[str, Any] | None) -> str | None:
    kind = task.get("kind") if isinstance(task, Mapping) else None
    return kind if kind in PROMPT_KINDS else None


def resolve_mode(mode: str | None, task: Mapping[str, Any] | None) -> str:
    """The task's kind is more specific than the session mode, so it wins when they disagree."""
    return _task_kind(task) or normalize_mode(mode)


def quiz_stage(task: Mapping[str, Any] | None) -> str:
    stage = (task or {}).get("stage")
    return stage if stage in QUIZ_STAGES else "dictate"


def system_prompt(profile: str, mode: str, language: str = "en") -> str:
    language = normalize_reply_language(language)
    if normalize_profile(profile) == "biology":
        return SYSTEM_PROMPT + BIOLOGY_SPANISH if language == "es" else SYSTEM_PROMPT
    kind = str(mode or "").strip().lower()
    return MATH_PROMPTS_BY_LANGUAGE[language][kind if kind in PROMPT_KINDS else "help"]


def _ink_text(ink_regions: list[InkRegion], *, math: bool = False, language: str = "en") -> str:
    spanish = normalize_reply_language(language) == "es"
    head = "Regiones de tinta detectadas en el pizarrón" if spanish else "Ink regions detected on the whiteboard"
    if not ink_regions:
        return f"{head}: {'ninguna' if spanish else 'none'}."
    # In math, "x=300" reads as the student's answer: the model copied box coordinates as "x = 300".
    box = "left={x} top={y} width={w} height={h}" if math else "box x={x} y={y} w={w} h={h}"
    center = "centro" if spanish else "center"
    lines = [f"{i}. {box.format(**r)}, {center}=({r['cx']}, {r['cy']})" for i, r in enumerate(ink_regions, 1)]
    return f"{head}:\n" + "\n".join(lines)


def _context_text(context: list[RetrievedChunk], *, math: bool = False, language: str = "en") -> str:
    spanish = normalize_reply_language(language) == "es"
    head = "Contexto de referencia" if spanish else "Reference context"
    # Math notes are worksheets and quiz banks: their answers would be read out or copied as the student's answer.
    # Stripped together: an answer key can run over several chunks of one file.
    stripped = strip_chunk_answers([(c.source, c.text) for c in context]) if math else [c.text for c in context]
    texts = [(c.source, text) for c, text in zip(context, stripped, strict=True) if text.strip()]
    if not texts:
        return f"{head}: {'ninguno' if spanish else 'none'}."
    source = "fuente" if spanish else "source"
    blocks = [f"[{i}] ({source}: {name})\n{text[:MAX_CHUNK_CHARS]}" for i, (name, text) in enumerate(texts, 1)]
    return f"{head}:\n" + "\n\n".join(blocks)


def crop_text(crop: Any, language: str = "en") -> str | None:
    """Where the photo sits on the board, when it is a crop of the inked part: targets stay in projector pixels."""
    roi = getattr(crop, "roi", None)
    if roi is None:
        return None
    x, y, w, h = (int(v) for v in roi)
    if normalize_reply_language(language) == "es":
        return (
            f"La foto muestra solo la parte del pizarrón con tinta, en píxeles del proyector: left={x} top={y} "
            f"width={w} height={h}. Las regiones de tinta y target_x / target_y van en píxeles del proyector, "
            "no de la foto."
        )
    return (
        f"The photo shows only the inked part of the board, in projector pixels: left={x} top={y} width={w} "
        f"height={h}. Ink regions and target_x / target_y are projector pixels, not photo pixels."
    )


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


_MIXED_RE = re.compile(r"(-?)(\d+)\s+(\d+)\s*/\s*(\d+)")


def answer_value(text: Any) -> Fraction | None:
    """The number in an answer like "x = 5", "-4", "2/3", "1 1/2" or "0.5"; None unless it is one plain number."""
    sides = str(text if text is not None else "").strip().lower().replace("−", "-").rstrip(".").split("=")
    if len(sides) > 2 or (len(sides) == 2 and not re.fullmatch(r"\s*[a-z]?\s*", sides[0])):
        return None
    value = sides[-1].strip()
    mixed = _MIXED_RE.fullmatch(value)
    try:
        if mixed:
            sign, whole, num, den = mixed.groups()
            return (-1 if sign else 1) * (int(whole) + Fraction(int(num), int(den)))
        return Fraction(value.replace(" ", "")) if value else None
    except (ValueError, ZeroDivisionError):
        return None


def _seed_problem(number: int, total: int) -> str:
    # Deterministic per number, so the prompt and evaluate's post-processing pick the same problem.
    return invent_problem(number, total, random.Random(number * 7919 + total)).problem


def expected_answer(task: Mapping[str, Any] | None) -> str | None:
    """The task's answer, or the one code works out from the problem (e.g. an invented problem)."""
    t = task or {}
    return _clean(t.get("answer")) or solve_problem(_clean(t.get("problem")))


def quiz_task_text(task: Mapping[str, Any] | None, language: str = "en") -> str:
    t = task or {}
    if normalize_reply_language(language) == "es":
        return _quiz_task_text_es(t)
    stage = quiz_stage(t)
    number = _as_int(t.get("number"), 1)
    total = _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
    problem, upcoming = _clean(t.get("problem")), _clean(t.get("next_problem"))
    head = f"Quiz stage: {stage}. Problem {number} of {total}: "
    if stage == "dictate":
        # With no worksheet problem a 3B model echoes the student instead of making one up: code picks it.
        problem = problem or _seed_problem(number, total)
        return f"{head}{problem}\nRead this problem out in words. Do not solve it."
    if stage == "hint":
        return (
            f"{head}{problem or 'the problem on the board'}\nThe student is stuck. Explain ONLY the next single step "
            "for this problem and ask them to write it. NEVER say the answer or the value of the letter."
        )
    lines = [head + (problem or "the problem on the board (see the conversation).")]
    answer = expected_answer(t)
    # A numeric answer is checked in code (check_quiz_answer) and left out here: shown it, a small model copies it
    # into board_answer instead of reading the board, and may say it aloud.
    if answer and answer_value(answer) is None:
        lines.append(f"Expected answer: {answer}. The student's answer is correct ONLY if its value is the same.")
    elif not answer:
        lines.append("Expected answer: not given, solve the problem yourself.")
    if upcoming:
        lines.append(f"Next problem: {upcoming}\nOnly if the answer is correct, also read the next problem in words.")
    elif number >= total:
        lines.append("This is the last problem: if the answer is correct, congratulate the student on finishing.")
    lines.append(
        "First copy the bottom line of the student's writing in the photo into board_answer, exactly as written "
        '(not a problem from this message); do not solve anything yourself. If the answer is wrong: "error", one '
        "hint, never the answer."
    )
    return "\n".join(lines)


def _quiz_task_text_es(t: Mapping[str, Any]) -> str:
    stage = quiz_stage(t)
    number = _as_int(t.get("number"), 1)
    total = _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
    problem, upcoming = _clean(t.get("problem")), _clean(t.get("next_problem"))
    head = f"Etapa del quiz: {QUIZ_STAGE_NAMES_ES[stage]}. Problema {number} de {total}: "
    if stage == "dictate":
        problem = problem or _seed_problem(number, total)
        return f"{head}{problem}\nLee este problema en voz alta con palabras, en español. No lo resuelvas."
    if stage == "hint":
        return (
            f"{head}{problem or 'el problema del pizarrón'}\nEl estudiante está atorado. Explica SOLO el siguiente "
            "paso de este problema y pídele que lo escriba. NUNCA digas la respuesta ni el valor de la letra."
        )
    lines = [head + (problem or "el problema del pizarrón (mira la conversación).")]
    answer = expected_answer(t)
    if answer and answer_value(answer) is None:
        lines.append(f"Respuesta esperada: {answer}. La respuesta del estudiante es correcta SOLO si vale lo mismo.")
    elif not answer:
        lines.append("Respuesta esperada: no se da, resuelve tú el problema.")
    if upcoming:
        lines.append(
            f"Siguiente problema: {upcoming}\nSolo si la respuesta es correcta, lee también el siguiente problema "
            "con palabras."
        )
    elif number >= total:
        lines.append("Este es el último problema: si la respuesta es correcta, felicita al estudiante por terminar.")
    lines.append(
        "Primero copia en board_answer la última línea de lo que escribió el estudiante en la foto, tal como está "
        '(no un problema de este mensaje); no resuelvas nada tú. Si la respuesta está mal: "error", una pista, '
        "nunca la respuesta."
    )
    return "\n".join(lines)


def learn_stage(task: Mapping[str, Any] | None) -> str:
    stage = (task or {}).get("stage")
    return stage if stage in LEARN_STAGES else "offer"


def learn_task_text(task: Mapping[str, Any] | None, language: str = "en") -> str:
    stage = learn_stage(task)
    if normalize_reply_language(language) == "es":
        topic = _clean((task or {}).get("topic")) or "este tema de matemáticas"
        todo = {
            "offer": f'Presenta la idea de {topic} en una o dos oraciones sencillas y luego pregunta: "{OFFER_QUESTION_ES}"',
            "example": f"El estudiante quiere que inventes uno: dicta con palabras UN problema fácil para empezar sobre "
            f"{topic} e invítalo a escribirlo e intentar el primer paso. No lo resuelvas.",
            "own": "El estudiante tiene su propio ejercicio: pídele que lo escriba en el pizarrón y que luego te pida "
            "ayuda.",
        }[stage]
        return f"Etapa de aprender: {LEARN_STAGE_NAMES_ES[stage]}. Tema: {topic}\n{todo}"
    topic = _clean((task or {}).get("topic")) or "this math topic"
    todo = {
        "offer": f'Introduce the idea of {topic} in one or two plain sentences, then ask: "{OFFER_QUESTION}"',
        "example": f"The student wants you to make one up: dictate ONE easy starter problem on {topic} in words "
        "and invite them to write it and try the first step. Do not solve it.",
        "own": "The student has their own exercise: ask them to write it on the board and then ask you for help.",
    }[stage]
    return f"Learn stage: {stage}. Topic: {topic}\n{todo}"


def _short(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= MAX_TURN_CHARS else text[: MAX_TURN_CHARS - 3].rstrip() + "..."


def turns_text(
    recent_turns: Sequence[Mapping[str, Any]] | None, speaker: str = "Student", language: str = "en"
) -> str | None:
    """The last few exchanges, oldest first, so follow-ups like "make one up" or "I did it" make sense."""
    spanish = normalize_reply_language(language) == "es"
    if spanish and speaker == "Student":
        speaker = "Estudiante"
    you = "Tú" if spanish else "You"
    lines: list[str] = []
    for turn in list(recent_turns or [])[-MAX_TURNS:]:
        if not isinstance(turn, Mapping):
            continue
        question, answer = _short(turn.get("q")), _short(turn.get("a"))
        lines += [f"{speaker}: {question}"] if question else []
        lines += [f"{you}: {answer}"] if answer else []
    if not lines:
        return None
    head = "Conversación hasta ahora (lo más antiguo primero)" if spanish else "Conversation so far (oldest first)"
    return f"{head}:\n" + "\n".join(lines)


_CHECK_RE = re.compile(
    r"\b(?:check|verify|correct|right|complete|wrong|missing|ok(?:ay)?|good|look\s+right|is\s+(?:this|it)\s+fine)\b",
    re.I,
)


def build_prompt(
    query: str,
    context: list[RetrievedChunk],
    ink_regions: list[InkRegion],
    projector_size: tuple[int, int],
    *,
    profile: str = "biology",
    mode: str = "help",
    task: Mapping[str, Any] | None = None,
    recent_turns: Sequence[Mapping[str, Any]] | None = None,
    language: str = "en",
    crop: Any = None,
) -> str:
    width, height = projector_size
    language = normalize_reply_language(language)
    spanish = language == "es"
    size = f"Projector size: {width}x{height} pixels (origin top-left)."
    where = crop_text(crop, language)
    if normalize_profile(profile) == "biology":
        parts = [f"User query: {query}", size, *([where] if where else []), _ink_text(ink_regions)]
        parts.append(_context_text(context))
        # A check is judged on the current board alone: shown an earlier verdict, the 3B model repeats it
        # (an earlier "not yet" failed a corrected chart, an earlier "complete" passed a broken one).
        history = None if _CHECK_RE.search(query or "") else turns_text(recent_turns, speaker="User")
        if history:
            parts.append(f"{history}\nEarlier turns are only for follow-up questions: judge the CURRENT board.")
        closing = "Answer with the JSON object only." + (" spoken_text in Spanish." if spanish else "")
        return "\n\n".join([*parts, closing])
    mode = resolve_mode(mode, task)
    labels, reminders = (MODE_LABELS_ES, MODE_REMINDERS_ES) if spanish else (MODE_LABELS, MODE_REMINDERS)
    head = "Modo" if spanish else "Mode"
    parts = [f"{head}: {labels[mode]}." + (f" {reminders[mode]}" if mode in reminders else "")]
    if mode == "quiz":
        parts.append(quiz_task_text(task, language))
    elif mode == "learn":
        parts.append(learn_task_text(task, language))
    history = turns_text(recent_turns, language=language)
    if spanish:
        size = f"Tamaño del proyector: {width}x{height} píxeles (origen arriba a la izquierda)."
        said, center = f"El estudiante dijo: {query}", f"Centro de la pantalla: ({width // 2}, {height // 2})."
        closing = (
            "Responde solo con el objeto JSON, en español. Escribe toda la matemática con palabras, también los "
            "números (cinco, no 5)."
        )
    else:
        said, center = f"Student said: {query}", f"Screen center: ({width // 2}, {height // 2})."
        closing = "Answer with the JSON object only. Write all math in words, numbers too (five, not 5)."
    parts += [
        *([history] if history else []),
        said,
        f"{size} {center}",
        *([where] if where else []),
        _ink_text(ink_regions, math=True, language=language),
        _context_text(context, math=True, language=language),
        closing,
    ]
    return "\n\n".join(parts)


def _reply_value(text: str | None, key: str) -> str | None:
    """A helper key of the model's reply (board_answer, board_problem): a string or a plain JSON number. The regex
    only rescues truncated JSON."""
    data = _extract_json_object(text or "")
    if data is not None:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        return _clean(value)
    match = re.search(rf'"{key}"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+(?:\.\d+)?))', text or "")
    if not match:
        return None
    if match.group(2) is not None:
        return match.group(2)
    try:
        return _clean(json.loads(f'"{match.group(1)}"'))
    except ValueError:
        return None


def board_answer(text: str | None) -> str | None:
    return _reply_value(text, "board_answer")


# ---- what a reply says, checked in code ------------------------------------------------------------------------

_NUMBER_WORDS = frozenset(
    re.findall(
        r"\w+",
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
        "half third fourth fifth sixth seventh eighth ninth tenth plus minus times over equals negative",
    )
)
_WORD_FORMS = {"halves": "half", "quarter": "fourth", "quarters": "fourth", "divided": "over"}
# Spanish, accents stripped: number words and fractions as mathspeak_es says them, operations under the English
# names so the checks below read both languages the same way ("más" -> "plus", "es igual a" -> "equals").
_NUMBER_WORDS_ES = frozenset(
    re.findall(
        r"\w+",
        "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciseis "
        "diecisiete dieciocho diecinueve veinte veintiuno veintidos veintitres veinticuatro veinticinco veintiseis "
        "veintisiete veintiocho veintinueve treinta cuarenta cincuenta sesenta setenta ochenta noventa cien ciento "
        "doscientos trescientos cuatrocientos quinientos seiscientos setecientos ochocientos novecientos mil "
        "medio tercio cuarto quinto sexto septimo octavo noveno decimo plus minus times over equals",
    )
)
_WORD_FORMS_ES = {
    "un": "uno",
    "una": "uno",
    "veintiun": "veintiuno",
    "mas": "plus",
    "menos": "minus",
    "negativo": "minus",
    "por": "times",
    "entre": "over",
    "sobre": "over",
    "igual": "equals",
}
# "por favor" is no product and "dividido entre" one division, not two.
_NOT_MATH_ES = re.compile(
    r"\bpor\s+(?:favor|eso|ejemplo|lo\s+tanto|fin|que|cierto|ahora|cada|ambos|los\s+dos|el\s+otro|la\s+otra|un\s+lado)\b"
    r"|\bentre\s+(?:ambos|los\s+dos|todos)\b|\bsobre\s+(?:el|la|los|las)\s+(?:pizarron|pizarra|tema|hoja)\b"
)
_DIVIDED_ES = re.compile(r"\bdividid[oa]s?\s+(?:entre|por|sobre)\b")
_NUMBER = (
    r"(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
)
# "un" / "una" are left out like English "a": "la respuesta es una fracción" gives nothing away.
_NUMBER_ES = (
    r"(?:\d|cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|"
    r"dieci\w+|veinte|veinti\w+|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien|ciento|mil)"
)
# "So x equals five.", "The answer is 5", "x should be seven": the value of the letter said as a result. Not a step
# such as "three x equals fifteen" (the letter has a number in front) and not a question.
_REVEAL_RE = re.compile(
    r"(?:^|[,;:]|\b(?:so|then|and|means|now|great|yes|correct|because)\b)\s*"
    r"(?:x|y|n|the\s+(?:answer|solution|value\s+of\s+[a-z])|your\s+answer)\s+"
    r"(?:=|equals|is(?:\s+equal\s+to)?|would\s+be|will\s+be|should\s+be|must\s+be)\s+"
    rf"(?:negative\s+|minus\s+)?{_NUMBER}"
)
# The same in Spanish, on accent-free text: "Entonces x es igual a cinco.", "La respuesta es cinco", "x vale siete".
_REVEAL_RE_ES = re.compile(
    r"(?:^|[,;:]|\b(?:asi\s+que|entonces|o\s+sea|por\s+lo\s+tanto|luego|y|es\s+decir|correcto|bien|si|porque|"
    r"ahora|queda|nos\s+queda|te\s+queda)\b)\s*"
    r"(?:x|n|ye|la\s+(?:respuesta|solucion)|el\s+valor\s+de\s+(?:la\s+)?(?:x|ye|n|letra|incognita)|tu\s+respuesta)\s+"
    r"(?:=|es\s+igual\s+a|es|vale|equivale\s+a|igual\s+a|seria|sera|debe\s+ser|deberia\s+ser|da|queda\s+en)\s+"
    rf"(?:menos\s+|negativo\s+)?{_NUMBER_ES}\b"
)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
# Sentences of text that may still hold notation: "3.5" is no sentence end, "Try 3.5 - x. Then..." is two.
_RAW_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _raw_sentences(text: str) -> list[str]:
    return [s.strip() for s in _RAW_SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]


def _spoken(text: str, language: str = "en") -> str:
    """The text as the voice will say it, so "x = 5" and "x equals five" compare the same."""
    to_speech = _speech_function(language)
    if to_speech is None:
        return text
    try:
        return str(to_speech(text))
    except Exception:
        return text


def _plain(text: str) -> str:
    """Lowercase without accents ("más" -> "mas"), for matching Spanish."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))


def _norm_words(text: str, language: str = "en") -> str:
    spoken = _plain(_spoken(text, language))
    if normalize_reply_language(language) == "es":
        spoken = _DIVIDED_ES.sub("entre", _NOT_MATH_ES.sub(" ", spoken))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", spoken).split())


def _math_seq(text: str, language: str = "en") -> list[str]:
    """The numbers and operations of a text as words, in order ("thirds" -> "third", "divided" -> "over")."""
    spanish = normalize_reply_language(language) == "es"
    words, forms = (_NUMBER_WORDS_ES, _WORD_FORMS_ES) if spanish else (_NUMBER_WORDS, _WORD_FORMS)
    out: list[str] = []
    for word in _norm_words(text, language).split():
        word = forms.get(word, word)
        if word.endswith("s") and word[:-1] in words:
            word = word[:-1]
        if word in words or word.isdigit():
            out.append(word)
    return out


def _math_words(text: str, language: str = "en") -> Counter[str]:
    return Counter(_math_seq(text, language))


def dictates_problem(spoken: str, like: str | None = None, language: str = "en") -> bool:
    """Whether a reply dictates a whole problem: an operation, and "equals" when it should be an equation."""
    said = set(_math_seq(spoken, language))
    if not said & {"plus", "minus", "times", "over"}:
        return False
    return "equals" in said or "=" not in (like or "")


def reads_back(spoken: str, problem: str | None, language: str = "en") -> bool:
    """Whether a reply says this exact problem (its numbers and operations in the same order)."""
    wanted, said = _math_seq(problem or "", language), _math_seq(spoken, language)
    n = len(wanted)
    return bool(wanted) and any(said[i : i + n] == wanted for i in range(len(said) - n + 1))


def reads_problem(spoken: str, problem: str | None, language: str = "en") -> bool:
    """Whether a reply reads the problem out: all of its numbers and operations are there (in words or not)."""
    wanted = _math_words(problem or "", language)
    return bool(wanted) and not (wanted - _math_words(spoken, language))


def reveals_value(spoken: str, language: str = "en") -> bool:
    """A sentence (not a question) that says what the letter equals: a final answer."""
    spanish = normalize_reply_language(language) == "es"
    text = _plain(_spoken(spoken, language)) if spanish else _spoken(spoken, language).lower()
    pattern = _REVEAL_RE_ES if spanish else _REVEAL_RE
    sentences = _SENTENCE_RE.findall(text)
    return any(pattern.search(s.strip(" ¿¡")) for s in sentences if not s.strip().endswith("?"))


_WORD_RE = re.compile(r"[^\W\d_]{3,}")


def _sentence_in_language(sentence: str, language: str) -> bool:
    from biobench_sar.lang import detect_language

    if language == "es":
        # A Spanish voice reading "Correct, great job!" is wrong even though no word of it is on the English list:
        # a sentence with real words must lean Spanish. Math alone ("x = 5") leans neither way and is fine.
        if len(_WORD_RE.findall(sentence)) < 2:
            return True
        return detect_language(sentence, default="en") == "es"
    # English replies ("Great job!") often use words on neither list: only plain Spanish counts as wrong.
    return len(sentence.split()) < 3 or detect_language(sentence, default="en") == "en"


def in_language(text: str, language: str = "en") -> bool:
    """Whether every sentence of a reply is in the reply language (or in neither, like bare math)."""
    language = normalize_reply_language(language)
    try:
        return all(_sentence_in_language(sentence, language) for sentence in _raw_sentences(text))
    except Exception:  # language detection missing or broken: never block a reply on it
        return True


def _similar(a: str, b: str, language: str = "en") -> float:
    a, b = _norm_words(a, language), _norm_words(b, language)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _is_echo(spoken: str, query: str | None, language: str = "en") -> bool:
    return bool(query) and _similar(spoken, str(query), language) >= ECHO_RATIO


def _repeated(spoken: str, recent_turns: Sequence[Mapping[str, Any]] | None, language: str = "en") -> str | None:
    """The earlier reply this one copies word for word (all of it, or nearly), if any."""
    words = _norm_words(spoken, language)
    for turn in reversed(list(recent_turns or [])[-MAX_TURNS:]):
        earlier = str(turn.get("a") or "") if isinstance(turn, Mapping) else ""
        before = _norm_words(earlier, language)
        similar = _similar(spoken, earlier, language) >= REPEAT_RATIO
        if before and (similar or (len(before.split()) >= 4 and before in words)):
            return earlier
    return None


# ---- deterministic task rules ---------------------------------------------------------------------------------


def _dictation(number: int, total: int, problem: str, language: str = "en") -> str:
    return text_for(language, "dictation", number=number, total=total, problem=problem)


def check_quiz_answer(
    action: AvatarAction,
    task: Mapping[str, Any] | None,
    answer_on_board: str | None,
    *,
    query: str | None = None,
    language: str = "en",
) -> AvatarAction:
    """Grade the quiz answer in code: a 3B model reads "x = 7" fine but often calls it correct anyway."""
    t = task or {}
    number, total = _as_int(t.get("number"), 1), _as_int(t.get("total"), 0)
    upcoming = _clean(t.get("next_problem"))
    expected, given = answer_value(expected_answer(t)), answer_value(answer_on_board)
    spoken, state = action.spoken_text, action.animation_state
    usable = (
        spoken.strip() not in ("", "...") and not _is_echo(spoken, query, language) and in_language(spoken, language)
    )
    correct = text_for(language, "correct")
    if expected is None:  # not a plain number (two solutions, a word answer): the model's verdict stands
        verdict = {"celebrating": "correct", "error": "wrong"}.get(state)
        if verdict == "wrong" and not usable:
            return _tutor(action, spoken_text=text_for(language, "wrong"), verdict=verdict)
        if verdict != "correct":
            return _tutor(action, verdict=verdict)
        praise = spoken if usable else correct
    elif given is None:
        # Nothing read off the board: never approve (the model praises anything) and never count it as a miss.
        return _tutor(action, spoken_text=text_for(language, "no_answer"), animation_state="idle", verdict=None)
    elif given != expected:
        keep = (
            state == "error"
            and usable
            and not reveals_value(spoken, language)
            and not reads_problem(spoken, upcoming, language)
        )
        wrong = spoken if keep else text_for(language, "wrong")
        return _tutor(action, spoken_text=wrong, animation_state="error", verdict="wrong")
    else:
        praise = spoken if state == "celebrating" and usable else correct
    if upcoming and not reads_problem(praise, upcoming, language):
        # The next turn checks it, so it must be read out now.
        praise += text_for(language, "next", number=number + 1, problem=upcoming)
    elif not upcoming and total and number >= total and praise == correct:
        praise += text_for(language, "last")
    return _tutor(action, spoken_text=praise, animation_state="celebrating", verdict="correct", dictated=upcoming)


def _quiz_dictation(
    action: AvatarAction, task: Mapping[str, Any], query: str | None, language: str = "en"
) -> AvatarAction:
    number, total = _as_int(task.get("number"), 1), _as_int(task.get("total"), DEFAULT_QUIZ_TOTAL)
    problem = _clean(task.get("problem")) or _seed_problem(number, total)
    spoken = action.spoken_text
    # The model often echoes "Quiz me." or reads an earlier turn: the student must hear exactly this problem.
    if (
        _is_echo(spoken, query, language)
        or not reads_problem(spoken, problem, language)
        or reveals_value(spoken, language)
        or not in_language(spoken, language)
    ):
        spoken = _dictation(number, total, problem, language)
    return _tutor(action, spoken_text=spoken, animation_state="idle", dictated=problem)


# "One half plus one fourth equals three fourths.": an arithmetic problem dictated together with its result.
# Equations are left alone: in "x plus three equals eight" the "equals eight" is part of the problem.
_SOLVED_RE = re.compile(
    r"^(?P<problem>.*\b(?:plus|minus|times|over|divided\s+by)\b.*?)\s*,?\s+"
    r"(?:equals|is\s+equal\s+to|is|makes|gives(?:\s+you)?)\s+[^.!?]+(?P<end>[.!?]*)$",
    re.I,
)
_LETTER_RE = re.compile(r"\b[a-z]\b(?<!\ba)(?<!\bi)", re.I)
# "Un medio más un cuarto es igual a tres cuartos." The same in Spanish ("por favor" is no product).
_SOLVED_RE_ES = re.compile(
    r"^(?P<problem>.*\b(?:m[aá]s|menos|por(?!\s+(?:favor|eso|ejemplo|lo\s+tanto|qu[eé]))|entre|sobre)\b.*?)\s*,?\s+"
    r"(?:es\s+igual\s+a|igual\s+a|es|son|da|nos\s+da|te\s+da|resulta(?:\s+en)?)\s+[^.!?]+(?P<end>[.!?]*)$",
    re.I,
)
# A letter of the problem; Spanish "a", "e", "o", "u", "y" are words, and the variable y is said "ye".
_LETTER_RE_ES = re.compile(r"(?<![\wáéíóúñ])(?:[b-df-hj-np-tv-xz]|ye)(?![\wáéíóúñ])", re.I)


def unsolved(spoken: str, language: str = "en") -> str:
    """A dictated problem without the result the model added to it (practice problems and lesson starters).
    Only a sentence that gave the result away is changed, spelled out; the others keep their notation."""
    spanish = normalize_reply_language(language) == "es"
    solved, letter = (_SOLVED_RE_ES, _LETTER_RE_ES) if spanish else (_SOLVED_RE, _LETTER_RE)
    out: list[str] = []
    for sentence in _raw_sentences(spoken):
        said = _spoken(sentence, language).strip()
        found = solved.match(said)
        if found and not letter.search(found.group("problem")):
            out.append(found.group("problem").rstrip(" ,") + (found.group("end") or "."))
        else:
            out.append(sentence)
    return " ".join(out) or spoken


_OFFER_ASKED_RE = re.compile(r"\b(?:make\s+one\s+up|exercise|one\s+for\s+you|problem\s+in\s+mind)\b", re.I)
_OFFER_ASKED_RE_ES = re.compile(r"\b(?:invent[eoa]\w*|ejercicio|en\s+mente|uno\s+para\s+ti)\b", re.I)


def _with_offer_question(spoken: str, language: str = "en") -> str:
    question = text_for(language, "offer")
    asked = _OFFER_ASKED_RE_ES if normalize_reply_language(language) == "es" else _OFFER_ASKED_RE
    spoken = spoken.strip()
    if spoken in ("", "..."):
        return question
    sentences = _raw_sentences(spoken)
    if sentences and sentences[-1].endswith("?") and asked.search(sentences[-1]):
        return spoken  # asked in its own words
    if sentences and sentences[-1].endswith("?"):
        sentences = sentences[:-1]  # one question at the end, not two in a row
    return " ".join([*sentences, question])


def apply_task_rules(
    action: AvatarAction,
    mode: str,
    task: Mapping[str, Any] | None,
    answer_on_board: str | None = None,
    *,
    query: str | None = None,
    language: str = "en",
) -> AvatarAction:
    """Make what the task fixes deterministic where a 3B model wavers: quiz grading and dictation, idle states,
    the offer question. A failed call is left as it is; any state the model picks is not a failure."""
    if getattr(action, "failed", False):
        return action
    kind = resolve_mode(mode, task)
    t = task or {}
    if kind == "quiz":
        stage = quiz_stage(t)
        if stage == "check":
            return check_quiz_answer(action, t, answer_on_board, query=query, language=language)
        if stage == "hint":
            return replace(action, animation_state="pointing")
        return _quiz_dictation(action, t, query, language)
    if kind == "learn" and learn_stage(t) == "offer":
        action = replace(action, spoken_text=_with_offer_question(action.spoken_text, language))
    if kind == "practice" or (kind == "learn" and learn_stage(t) == "example"):
        # The student solves it, not the voice.
        action = replace(action, spoken_text=unsolved(action.spoken_text, language))
    # Dictating or offering is not about one drawing; at temperature 0.6 the model still picks "pointing"/"walking".
    return replace(action, animation_state="idle") if kind in ("practice", "learn") else action


def _starter(topic: str | None) -> str:
    if re.search(r"fraction|fracci", _plain(topic or "")):
        return "1/5 + 2/5"
    return "x + 3 = 8"


def _starter_missing(action: AvatarAction, task: Mapping[str, Any] | None, language: str) -> bool:
    """A lesson starter on fractions or equations that dictates no problem at all (a story about apples instead):
    the student has nothing to write. Other topics (square roots, angles) may start without an operation."""
    if learn_stage(task) != "example":
        return False
    topic = _plain(str((task or {}).get("topic") or ""))
    if not re.search(r"fraction|fracci|equation|ecuaci", topic):
        return False
    return not dictates_problem(action.spoken_text, None, language)


def fallback_text(kind: str, task: Mapping[str, Any] | None, has_ink: bool, language: str = "en") -> str:
    """What to say when the model's reply is unusable even after a retry."""
    t = task or {}
    if kind == "learn":
        topic = _clean(t.get("topic")) or text_for(language, "this_topic")
        stage = learn_stage(t)
        if stage == "offer":
            return text_for(language, "learn_offer", topic=topic, offer=text_for(language, "offer"))
        if stage == "example":
            return text_for(language, "learn_example", problem=_starter(topic))
        return text_for(language, "learn_own")
    if kind == "practice":
        return text_for(language, "try_this", problem=_seed_problem(3, 6))
    if kind == "quiz" or has_ink:
        return text_for(language, "step")
    return text_for(language, "write_problem")


RETRY_NOTES = {
    "en": {
        "echo": "Your last reply only repeated the student's words. Answer what they asked instead.",
        "language": "Your last reply was not in English. Answer in English.",
        "read_back": "Your last reply read the board's problem ({board}) back. Dictate a NEW problem with new numbers.",
        "not_whole": "Your last reply did not dictate a whole new problem like {board}. Say all of it in words.",
        "copied": 'Your last reply copied an earlier one ("{earlier}"). Answer the student\'s latest message.',
        "reveal": "Your last reply gave away the value of the letter. Explain only the next step, never the answer.",
        "starter": "Your last reply did not dictate a problem. Dictate ONE easy problem on {topic} in words, with its "
        "numbers and operation, not a story.",
    },
    "es": {
        "echo": "Tu respuesta anterior solo repitió las palabras del estudiante. Responde lo que preguntó.",
        "language": "Tu respuesta anterior no estaba en español. Responde en español de Latinoamérica, de tú.",
        "read_back": "Tu respuesta anterior volvió a leer el problema del pizarrón ({board}). Dicta un problema NUEVO "
        "con números nuevos.",
        "not_whole": "Tu respuesta anterior no dictó completo un problema nuevo como {board}. Dilo todo con palabras.",
        "copied": 'Tu respuesta anterior copió una anterior ("{earlier}"). Responde al último mensaje del estudiante.',
        "reveal": "Tu respuesta anterior dio el valor de la letra. Explica solo el siguiente paso, nunca la respuesta.",
        "starter": "Tu respuesta anterior no dictó un problema. Dicta UN problema fácil de {topic} con palabras, con "
        "sus números y su operación, no una historia.",
    },
}


def _retry_note(
    action: AvatarAction,
    kind: str,
    task: Mapping[str, Any] | None,
    query: str,
    recent_turns: Any,
    board: str | None = None,
    language: str = "en",
) -> str | None:
    """Why a finished math reply must be asked for again, or None. Quiz dictation and grading are fixed in code."""
    if kind == "quiz" and quiz_stage(task) != "hint":
        return None
    notes = RETRY_NOTES[normalize_reply_language(language)]
    spoken = action.spoken_text
    if _is_echo(spoken, query, language):
        return notes["echo"]
    if not in_language(getattr(action, "display_text", None) or spoken, language):
        return notes["language"]
    if kind == "practice" and board and reads_back(spoken, board, language):
        return notes["read_back"].format(board=board)
    if kind == "practice" and board and solve_problem(board) and not dictates_problem(spoken, board, language):
        return notes["not_whole"].format(board=board)
    earlier = _repeated(spoken, recent_turns, language)
    if earlier:
        return notes["copied"].format(earlier=_short(earlier))
    if kind in ("help", "quiz") and reveals_value(spoken, language):
        return notes["reveal"]
    if kind == "learn" and _starter_missing(action, task, language):
        return notes["starter"].format(topic=_clean((task or {}).get("topic")))
    return None


def _salvage(
    action: AvatarAction,
    kind: str,
    task: Mapping[str, Any] | None,
    query: str,
    recent_turns: Any,
    has_ink: bool,
    board: str | None = None,
    language: str = "en",
) -> AvatarAction:
    """Keep the sentences that are fine (not an echo, a copy, the answer or the wrong language); otherwise use the
    task's fallback. Works on the text before mathspeak, so what is kept keeps its notation on screen."""
    text = getattr(action, "display_text", None) or action.spoken_text
    spoken = action.spoken_text
    if kind == "practice" and board:
        off = reads_back(spoken, board, language) or not dictates_problem(spoken, board, language)
        variant = similar_problem(board) if off or not in_language(text, language) else None  # same shape, new numbers
        if variant:
            return replace(action, spoken_text=text_for(language, "try_this", problem=variant))
    earlier = [str(t.get("a") or "") for t in list(recent_turns or []) if isinstance(t, Mapping)]

    def copied(sentence: str) -> bool:
        words = _norm_words(sentence, language)
        return any(
            _similar(sentence, e, language) >= REPEAT_RATIO
            or (len(words.split()) >= 4 and words in _norm_words(e, language))
            for e in earlier
            if e
        )

    kept = [
        s
        for s in _raw_sentences(text)
        if not _is_echo(s, query, language)
        and not reveals_value(s, language)
        and not copied(s)
        and in_language(s, language)
    ]
    out = " ".join(kept) if kept else fallback_text(kind, task, has_ink, language)
    if kind == "learn" and _starter_missing(replace(action, spoken_text=_spoken(out, language)), task, language):
        out = fallback_text(kind, task, has_ink, language)
    if kind == "learn" and learn_stage(task) == "offer":
        out = _with_offer_question(out, language)
    return replace(action, spoken_text=out)


def _speech_function(language: str = "en") -> Callable[[str], str] | None:
    """mathspeak.to_speech, or mathspeak_es.to_speech_es for Spanish; None when it cannot be imported."""
    spanish = normalize_reply_language(language) == "es"
    try:
        if spanish:
            from biobench_sar.llm.mathspeak_es import to_speech_es as to_speech
        else:
            from biobench_sar.llm.mathspeak import to_speech
    except ImportError:
        return None
    except Exception as exc:  # a broken module must not break evaluate, which never raises
        logger.warning("mathspeak import failed, speaking the raw text: %s", exc)
        return None
    return to_speech


def _math_to_speech() -> Callable[[str], str] | None:
    return _speech_function("en")


def speakable(action: AvatarAction, language: str = "en") -> AvatarAction:
    """Spell out any notation the model left in (2/3, x^2, 3x+5=12) so the TTS voice reads it naturally, in the reply
    language. The text before that stays on the action as display_text: the screen shows the notation."""
    raw = action.spoken_text
    spoken = raw
    to_speech = _speech_function(language)
    if to_speech is not None:
        try:
            spoken = str(to_speech(raw)).strip() or raw
        except Exception as exc:
            logger.warning("mathspeak failed, speaking the raw text: %s", exc)
    return _tutor(action, spoken_text=spoken, display_text=raw)


POINTED_STATES = frozenset({"pointing", "celebrating", "error"})


def _crop_target(action: AvatarAction, text: str | None, crop: Any, projector_size: tuple[int, int]) -> AvatarAction:
    """Targets in projector pixels when the model answered in pixels of the crop image instead.

    A target inside the crop's board area is taken as projector pixels (what the prompt asks for). One outside it but
    inside the crop image's own size is read as a crop pixel and mapped back with InkCrop.to_projector. Idle and
    walking targets are the screen center by rule, so they are never moved.
    """
    roi, size = getattr(crop, "roi", None), getattr(crop, "size", None)
    if roi is None or size is None or action.animation_state not in POINTED_STATES:
        return action
    data = _extract_json_object(text or "") or {}
    try:
        x, y = float(data.get("target_x")), float(data.get("target_y"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return action
    if not (math.isfinite(x) and math.isfinite(y)):
        return action
    rx, ry, rw, rh = roi
    if rx <= x <= rx + rw and ry <= y <= ry + rh:
        return action
    if not (0 <= x <= size[0] and 0 <= y <= size[1]):
        return action
    px, py = crop.to_projector(x, y)
    width, height = projector_size
    tx, ty = max(0, min(width - 1, round(px))), max(0, min(height - 1, round(py)))
    return replace(action, target_x=tx, target_y=ty)


class RawReply(NamedTuple):
    """The model's answer before evaluate's math post-processing."""

    action: AvatarAction
    board_answer: str | None
    failed: bool = False
    board_problem: str | None = None


class VLMClient:
    def __init__(
        self,
        backend: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 90,
        *,
        max_image_side: int = 1024,
        profile: str = "biology",
    ) -> None:
        self.backend = backend.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_image_side = max_image_side
        self.profile = normalize_profile(profile)
        self.session = requests.Session()

    def warmup(self) -> None:
        """Load the model into memory ahead of the first question. Never raises."""
        if self.backend != "ollama":
            return
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1},
            }
            self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout).raise_for_status()
        except Exception as exc:
            logger.warning("VLM warmup failed: %s", exc)

    def _prepare_image(self, image_jpeg: bytes | None) -> bytes | None:
        if not image_jpeg:
            return image_jpeg
        try:
            img = cv2.imdecode(np.frombuffer(image_jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return image_jpeg
            h, w = img.shape[:2]
            scale = self.max_image_side / max(h, w)
            if scale >= 1:
                return image_jpeg
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes() if ok else image_jpeg
        except Exception as exc:
            logger.warning("Image downscale failed, sending original: %s", exc)
            return image_jpeg

    def evaluate(
        self,
        query: str,
        image_jpeg: bytes | None,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
        *,
        mode: str = "help",
        task: Mapping[str, Any] | None = None,
        recent_turns: Sequence[Mapping[str, Any]] | None = None,
        language: str = "en",
        crop: Any = None,
    ) -> AvatarAction:
        """Never raises. A failed call comes back as a TutorAction with failed=True.

        language: "en" or "es", the language of the reply (prompts, code's own texts and mathspeak follow it).
        crop: an ink_crop.InkCrop when image_jpeg is a high-res crop of the inked part of the board: it is sent
        as it is (never shrunk again), the prompt says where it sits, and targets stay in projector pixels.
        """
        task = task if isinstance(task, Mapping) else None
        language = normalize_reply_language(language)
        extra = {"language": language, "crop": crop}
        args = (query, image_jpeg, context, ink_regions, projector_size, mode, task, recent_turns)
        reply = self._evaluate(*args, **extra)
        if reply.failed or self.profile != "math":
            return reply.action
        try:
            kind = resolve_mode(mode, task)
            t = task or {}
            if kind == "quiz" and quiz_stage(t) == "dictate" and not _clean(t.get("problem")):
                number, total = _as_int(t.get("number"), 1), _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
                task = {**t, "problem": _seed_problem(number, total)}
                args = (query, image_jpeg, context, ink_regions, projector_size, mode, task, recent_turns)
            action, board = self._finish(reply, mode, task, query, language), reply.board_problem
            note = None
            if self.backend != "mock":
                note = _retry_note(action, kind, task, query, recent_turns, board, language)
            if note is None:
                return action
            logger.info("asking the model again: %s", note)
            retry = self._evaluate(*args, note=note, **extra)
            if not retry.failed:
                again, board = self._finish(retry, mode, task, query, language), retry.board_problem or board
                if _retry_note(again, kind, task, query, recent_turns, board, language) is None:
                    return again
                action = again
            salvaged = _salvage(action, kind, task, query, recent_turns, bool(ink_regions), board, language)
            return speakable(salvaged, language)
        except Exception as exc:  # evaluate must never raise
            logger.exception("Math post-processing failed, using the raw answer: %s", exc)
            return reply.action

    def _finish(
        self, reply: RawReply, mode: str, task: Mapping[str, Any] | None, query: str, language: str = "en"
    ) -> AvatarAction:
        ruled = apply_task_rules(reply.action, mode, task, reply.board_answer, query=query, language=language)
        return speakable(ruled, language)

    def _evaluate(
        self,
        query: str,
        image_jpeg: bytes | None,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
        mode: str = "help",
        task: Mapping[str, Any] | None = None,
        recent_turns: Sequence[Mapping[str, Any]] | None = None,
        note: str | None = None,
        *,
        language: str = "en",
        crop: Any = None,
    ) -> RawReply:
        """The raw model answer and the board_answer it read, before evaluate's math post-processing."""
        language = normalize_reply_language(language)
        try:
            mode = resolve_mode(mode, task)
            if self.backend == "mock":
                text = self._mock(query, context, ink_regions, projector_size, mode, task, language)
                return RawReply(parse_vlm_response(text, projector_size), board_answer(text))
            prompt = build_prompt(
                query,
                context,
                ink_regions,
                projector_size,
                profile=self.profile,
                mode=mode,
                task=task,
                recent_turns=recent_turns,
                language=language,
                crop=crop,
            )
            if note:
                prompt += f"\n\n{note}"
            system = system_prompt(self.profile, mode, language)
            temperature = MODE_TEMPERATURE.get(mode, TEMPERATURE) if self.profile == "math" else TEMPERATURE
            temperature = RETRY_TEMPERATURE if note else temperature
            # A crop is already the detail the model needs: shrinking it again would blur the decimal points away.
            image_jpeg = image_jpeg if crop is not None else self._prepare_image(image_jpeg)
            if self.backend == "ollama":
                text = self._ollama(prompt, image_jpeg, system, temperature)
            elif self.backend == "openai":
                text = self._openai(prompt, image_jpeg, system, temperature)
            else:
                return RawReply(failed_action(f"Unknown VLM backend '{self.backend}'.", projector_size), None, True)
            if _extract_json_object(text or "") is None:
                logger.error("VLM reply is not JSON: %.200s", text)
                failed = _tutor(parse_vlm_response(text, projector_size), failed=True)
                return RawReply(replace(failed, spoken_text=text_for(language, "not_json")), None, True)
            action = _crop_target(parse_vlm_response(text, projector_size), text, crop, projector_size)
            return RawReply(action, board_answer(text), False, _reply_value(text, "board_problem"))
        except requests.Timeout:
            logger.error("VLM request timed out (backend=%s)", self.backend)
            key = "timeout"
        except requests.RequestException as exc:
            logger.error("VLM request failed: %s", exc)
            key = "unreachable"
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            logger.error("Bad VLM payload: %s", exc)
            key = "bad_reply"
        except Exception as exc:  # evaluate must never raise
            logger.exception("Unexpected VLM failure: %s", exc)
            key = "unexpected"
        return RawReply(failed_action(text_for(language, key), projector_size), None, True)

    def _ollama(
        self, prompt: str, image_jpeg: bytes | None, system: str = SYSTEM_PROMPT, temperature: float = TEMPERATURE
    ) -> str:
        user: dict[str, object] = {"role": "user", "content": prompt}
        if image_jpeg:
            user["images"] = [base64.b64encode(image_jpeg).decode("ascii")]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user],
            "stream": False,
            "format": "json",
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": NUM_PREDICT},
        }
        resp = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])

    def _openai(
        self, prompt: str, image_jpeg: bytes | None, system: str = SYSTEM_PROMPT, temperature: float = TEMPERATURE
    ) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if image_jpeg:
            b64 = base64.b64encode(image_jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        path = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self.session.post(f"{self.base_url}{path}", json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])

    def _mock(
        self,
        query: str,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
        mode: str = "help",
        task: Mapping[str, Any] | None = None,
        language: str = "en",
    ) -> str:
        """An offline stand-in. For math tasks it plays a model that follows the task (and grades every quiz answer
        as correct), so the whole tutor flow, quiz included, can be clicked through without a model."""
        width, height = projector_size
        center = {"target_x": width // 2, "target_y": height // 2}
        largest = max(ink_regions, key=lambda i: i["w"] * i["h"]) if ink_regions else None
        at_ink = {"target_x": largest["cx"], "target_y": largest["cy"]} if largest else center
        t = task or {}
        reply: dict[str, Any] | None = None
        if self.profile == "math":
            if mode == "quiz":
                stage = quiz_stage(t)
                number, total = _as_int(t.get("number"), 1), _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
                problem = _clean(t.get("problem")) or _seed_problem(number, total)
                if stage == "dictate":
                    spoken = _dictation(number, total, problem, language)
                    reply = {"spoken_text": spoken, "animation_state": "idle", **center}
                elif stage == "hint":
                    reply = {"spoken_text": text_for(language, "step"), "animation_state": "pointing", **at_ink}
                else:
                    answer = expected_answer(t) or ""
                    reply = {"board_answer": answer, "spoken_text": text_for(language, "correct"), **at_ink}
                    reply["animation_state"] = "celebrating"
            elif mode in ("learn", "practice"):
                spoken = fallback_text(mode, t, bool(ink_regions), language)
                reply = {"spoken_text": spoken, "animation_state": "idle", **center}
        if reply is None:
            spanish = language == "es"
            heard = f'Escuché: "{query}".' if spanish else f'I heard: "{query}".'
            source = context[0].source if context else ""
            cite = (f" Según {source}." if spanish else f" According to {source}.") if context else ""
            if largest:
                look = (
                    "Este dibujo parece el elemento principal."
                    if spanish
                    else "This drawing looks like the main element."
                )
                reply = {"spoken_text": f"{heard}{cite} {look}", "animation_state": "pointing", **at_ink}
            else:
                none = "Todavía no veo ningún dibujo." if spanish else "I don't see any drawing yet."
                reply = {"spoken_text": f"{heard}{cite} {none}", "animation_state": "idle", **center}
        return json.dumps(reply)
