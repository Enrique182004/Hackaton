"""Math tutor state: the current mode, a running quiz, a topic the student asked to learn, and recent turns."""

from __future__ import annotations

import logging
import random
import re
import threading
import unicodedata
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

from biobench_sar.rag.quiz_bank import QuizProblem, invent_problem

log = logging.getLogger(__name__)

MODES = ("help", "practice", "quiz")
DEFAULT_QUIZ_LENGTH = 10  # problems in a quiz without worksheet problems (they are invented, see start_quiz)
MAX_TURNS = 3
MAX_TURN_CHARS = 400
MAX_TOPIC_WORDS = 8


def _norm(text: str) -> str:
    text = (text or "").lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


_STOP_QUIZ = _compile(
    r"\b(?:stop|end|quit|exit|cancel|finish|leave)\s+(?:the\s+|this\s+|my\s+)?quiz\b",
    r"\bno\s+more\s+quiz",
    r"\bquiz\s+(?:is\s+)?over\b",
    r"\bdone\s+with\s+(?:the\s+|this\s+|my\s+)?quiz",
    r"\bstop\s+quizzing\b",
)
_QUIZ = _compile(
    r"\bquiz\s+me\b",
    r"\b(?:start|begin|do|take|give\s+me|let's\s+do|let's\s+start|lets\s+do|lets\s+start)\s+"
    r"(?:a\s+|the\s+|another\s+|my\s+)?(?:math\s+|new\s+)?quiz\b",
    r"\b(?:want|have|need|try)\s+(?:a\s+|another\s+)(?:math\s+|new\s+|short\s+|little\s+)?quiz\b",
    r"\bquiz\s+mode\b",
    r"\btest\s+me\b",
)
_PRACTICE = _compile(
    r"\bsimilar\s+(?:problem|one|exercise|question|equation)s?\b",
    r"\b(?:another|one\s+more|a\s+new)\s+(?:one\s+|problem\s+|exercise\s+|question\s+|equation\s+)?like\s+(?:this|that|it)\b",
    r"\bmore\s+practice\b",
    r"\bpractice\s+(?:mode|problems?)\b",
    r"\blet'?s\s+practi[cs]e\b",
    r"\b(?:want|like|need)\s+to\s+practi[cs]e\b",
    r"^practi[cs]e$",
)
_REPEAT = _compile(
    r"\b(?:repeat|say|read)\s+(?:that|it|the\s+(?:problem|question|equation))\s+again\b",
    r"\brepeat\s+(?:that|it|the\s+(?:problem|question|equation)|please)\b",
    r"^(?:repeat|again|pardon|sorry|what)$",
    r"\bwhat\s+was\s+the\s+(?:problem|question|equation)\b",
    r"\b(?:didn't|did\s+not)\s+(?:hear|get|catch)\s+(?:that|it|the\s+(?:problem|question))\b",
)
_HELP = _compile(
    r"\bhelp\b",
    r"\bnext\s+step\b",
    r"\b(?:i'm|im|i\s+am)\s+stuck\b",
    r"\bstuck\b",
    r"\bwhat(?:'s|\s+is|\s+do\s+i\s+do)?\s+next\b",
    r"\bhint\b",
)


def detect_mode(text: str) -> str | None:
    """The mode a student's words ask for, or None to stay in the current one."""
    norm = _norm(text)
    if _any(_STOP_QUIZ, norm):
        return "help"
    if _any(_QUIZ, norm):
        return "quiz"
    if _any(_PRACTICE, norm):
        return "practice"
    if _any(_HELP, norm):
        return "help"
    return None


def is_quiz_stop(text: str) -> bool:
    """An explicit "stop the quiz": plain "help"/"I'm stuck" during a quiz asks for a hint, not to quit."""
    return _any(_STOP_QUIZ, _norm(text))


# Mid-quiz, these ask to stop too; the assistant asks to confirm first, so a plain "stop" is enough.
_STOP_INTENT = _compile(
    r"^(?:please\s+|ok\s+|okay\s+|can\s+we\s+|let's\s+|lets\s+)?(?:stop|quit|exit|cancel|end\s+it|stop\s+it)"
    r"(?:\s+(?:please|now|here|it|this|that|for\s+now|for\s+today))*$",
    r"\b(?:i|we)\s+(?:want|wanna|would\s+like|need|have)\s+to\s+(?:stop|quit|end\s+this|leave)\b",
    r"\b(?:i|we)\s+(?:don't|dont|do\s+not)\s+want\s+to\s+(?:do\s+this|play|continue|keep\s+going)\b",
    r"\blet'?s\s+(?:stop|quit)\b",
    r"\b(?:can|could)\s+we\s+(?:stop|quit)\b",
    r"\bi\s+quit\b",
)


def is_stop_intent(text: str) -> bool:
    """Wanting to end a running quiz: "stop the quiz", "stop", "I want to stop", "let's quit". Plain "help",
    "I'm stuck" and "I'm done" (with the answer) are not."""
    norm = _norm(text)
    return _any(_STOP_QUIZ, norm) or _any(_STOP_INTENT, norm)


_YES_WORDS = (
    r"(?:yes|yeah|yep|yup|sure|ok|okay|alright|all\s+right|of\s+course|definitely|"
    r"do\s+it|go\s+ahead|i\s+do|i'm\s+sure|im\s+sure|i\s+am\s+sure|sounds\s+good|why\s+not)"
)
# Agreeing without saying to what: "please", "fine", "bueno" (Mexican Spanish's "well..."). With a clear yes they
# still agree ("yes please", "sí, bueno"); alone they never end a quiz, which would lose its progress.
_SOFT_YES_WORDS = r"(?:please|fine)"
_YES = re.compile(
    rf"^(?:{_SOFT_YES_WORDS}\s+)*(?:{_YES_WORDS})"
    rf"(?:\s+(?:{_YES_WORDS}|{_SOFT_YES_WORDS}|thanks|thank\s+you|stop|stop\s+it|end\s+it|i\s+want\s+to\s+stop))*$"
)
_SOFT_YES = re.compile(rf"^{_SOFT_YES_WORDS}(?:\s+{_SOFT_YES_WORDS})*$")
_YES_WORDS_ES = (
    r"(?:si|claro|dale|ok|okey|okay|va|sale|orale|andale|de\s+acuerdo|esta\s+bien|"
    r"si\s+quiero|claro\s+que\s+si|seguro|simon|sip|obvio)"
)
_SOFT_YES_WORDS_ES = r"(?:bueno|pues|por\s+favor|porfa)"
_YES_ES = re.compile(
    rf"^(?:{_SOFT_YES_WORDS_ES}\s+)*(?:{_YES_WORDS_ES})(?:\s+(?:{_YES_WORDS_ES}|{_SOFT_YES_WORDS_ES}|gracias|termina|"
    r"terminalo|terminar|para|paralo|ya\s+no|quiero\s+terminar))*$"
)
_SOFT_YES_ES = re.compile(rf"^{_SOFT_YES_WORDS_ES}(?:\s+{_SOFT_YES_WORDS_ES})*$")


def is_yes(text: str) -> bool:
    """A plain yes ("yes", "sure", "ok", "yeah, please"): nothing else in it, so "ok, keep going" is not one, and a
    lone "please" or "fine" is not one either."""
    return bool(_YES.match(_norm(text)))


def is_repeat_request(text: str) -> bool:
    """ "Can you repeat the problem?": read the current quiz problem again instead of grading the board."""
    return _any(_REPEAT, _norm(text))


# "learn ..." / "teach me ..." say what they want outright; the how-to forms must also name a math topic,
# or "how do I solve this?" would turn a question about the board into a lesson.
_LEARN_TRUSTED = _compile(r"\blearn(?:\s+about)?\s+(?P<rest>.+)", r"\bteach\s+me(?:\s+about)?\s+(?P<rest>.+)")
_LEARN_GUARDED = _compile(
    r"\bshow\s+me\s+how\s+to\s+(?P<verb>.+)",
    r"\bhow\s+(?:do|can|should|would|could)\s+(?:i|you|we|one|people)\s+(?P<verb>.+)",
    r"\bhow\s+to\s+(?P<verb>.+)",
    r"\bexplain(?:\s+to\s+me)?\s+(?P<rest>.+)",
    # "what is a fraction?", "I don't understand fractions", "help me with fractions", "tell me about slope"
    r"\bhow\s+(?:do|does)\s+(?P<rest>.+?)\s+work\b",
    r"^(?:so\s+|um\s+|ok\s+|okay\s+)?what(?:'s|\s+is|\s+are)\s+(?P<rest>.+)",
    r"\b(?:don't|dont|do\s+not)\s+(?:really\s+)?(?:understand|get)\s+(?P<rest>.+)",
    r"\b(?:want|need|like|trying|try)\s+to\s+understand\s+(?P<rest>.+)",
    r"\bhelp(?:\s+me)?\s+(?:with|understand|learn)\s+(?P<rest>.+)",
    r"\btell\s+me\s+about\s+(?P<rest>.+)",
)
# "help me with fractions" names the help mode and a topic: a lesson on an empty board, help with the ink otherwise.
_HELP_WITH_TOPIC = re.compile(r"\bhelp(?:\s+me)?\s+(?:with|understand|learn)\s+")
_MATH_TOPIC = re.compile(
    r"\b(?:fraction|equation|decimal|percent|ratio|proportion|exponent|power|square\s+root|root|radical|algebra|"
    r"variable|expression|inequalit|polynomial|factor|integer|negative\s+number|whole\s+number|number\s+line|"
    r"graph|slope|intercept|function|angle|area|perimeter|volume|triangle|circle|rectangle|mean|median|average|"
    r"probability|prime|multiple|divisor|denominator|numerator|division|multiplication|addition|subtraction|"
    r"times\s+table|order\s+of\s+operations|pemdas|distributive|like\s+terms|quadratic|pythagor|geometry|"
    r"word\s+problem|long\s+division|mixed\s+number|place\s+value|rounding|scientific\s+notation|"
    r"absolute\s+value|coordinate|linear|system\s+of)\w*"
)
_DEIXIS = re.compile(r"\b(?:this|that|these|those|it|here|there|board|above|below|mine)\b")
# "the equation", "my answer", "the next step", "the left side": the student's own work, not a topic. A named
# concept ("the order of operations", "the quadratic formula") is still a topic.
_BOARD_OBJECT = re.compile(
    r"\b(?:the|my|our|your)\s+(?:(?:next|last|first|second|third|other|left|right|top|bottom|same|whole|final|"
    r"current|new)\s+)?(?:equation|expression|problem|exercise|question|fraction|answer|step|side|term|variable|"
    r"number|sum|result|solution|bracket|parenthes\w*|denominator|numerator|work|mistake|x|y)s?\b"
    r"|\bnext\s+step\b|\bstep\s+(?:one|two|three|\d+)\b"
)
# An explicit problem ("3x + 5 = 20", "three x plus five equals twenty", "one half and one third") is a request
# for step-by-step help with it, not for a lesson.
_EXPLICIT = re.compile(
    r"\d|\b(?:equals|plus|minus|times|divided\s+by|squared|cubed)\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:halves|half|thirds?|fourths?|quarters?|fifths?|"
    r"sixths?|sevenths?|eighths?|ninths?|tenths?)\b"
)
_LEADING_FILLER = re.compile(
    r"^(?:more\s+about|a\s+bit\s+about|a\s+little\s+about|about|the\s+basics\s+of|some|a|an)\s+"
)
_TRAILING_FILLER = re.compile(
    r"(?:^|\s+)(?:please|now|today|thanks|thank\s+you|again|for\s+me|to\s+me|with\s+me|right\s+now)$"
)
_IRREGULAR_GERUNDS = {"do": "doing", "be": "being", "see": "seeing", "plot": "plotting", "get": "getting"}
_IRREGULAR_GERUNDS |= {"set": "setting", "put": "putting", "cut": "cutting", "sum": "summing"}


def _gerund(verb: str) -> str:
    if verb in _IRREGULAR_GERUNDS:
        return _IRREGULAR_GERUNDS[verb]
    if verb.endswith("ing"):
        return verb
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "ye", "oe")):
        return verb[:-1] + "ing"
    return verb + "ing"


def _about_the_board(rest: str) -> bool:
    return bool(_DEIXIS.search(rest) or _BOARD_OBJECT.search(rest) or _EXPLICIT.search(rest))


def _topic(rest: str, *, starts_with_verb: bool) -> str | None:
    rest = _LEADING_FILLER.sub("", rest.strip())
    for prefix in ("how to ", "to "):
        if rest.startswith(prefix):
            rest, starts_with_verb = rest[len(prefix) :], True
    while (trimmed := _TRAILING_FILLER.sub("", rest)) != rest:
        rest = trimmed
    # Checked on the whole phrase: cut to MAX_TOPIC_WORDS first, "... the equation on the board" lost "board".
    if _about_the_board(rest):
        return None
    words = rest.split()[:MAX_TOPIC_WORDS]
    if not words:
        return None
    if starts_with_verb:
        words[0] = _gerund(words[0])
    return " ".join(words)


def detect_learn_request(text: str, *, board_in_use: bool = False) -> str | None:
    """The topic of a general "I want to learn X" / "teach me X" / "how do I X?" request, else None.

    Never for the student's own work: "this", "the equation", "my answer", "the next step" or an explicit problem
    ("how do I solve 3x + 5 = 20"). The how-to and "explain" forms are also skipped when `board_in_use` (ink on the
    board or a quiz running) or when they ask for a mode ("explain the next step" is help): then they are about
    the board. "learn" / "teach me" say it outright and still start a lesson.
    """
    norm = _norm(text)
    for pattern in _LEARN_TRUSTED:
        found = pattern.search(norm)
        if found:
            return _topic(found.group("rest"), starts_with_verb=False)
    mode = detect_mode(text)
    if board_in_use or (mode is not None and not (mode == "help" and _HELP_WITH_TOPIC.search(norm))):
        return None
    for pattern in _LEARN_GUARDED:
        found = pattern.search(norm)
        if not found:
            continue
        groups = found.groupdict()
        rest = groups.get("verb") or groups.get("rest") or ""
        if not _MATH_TOPIC.search(rest):
            return None
        return _topic(rest, starts_with_verb="verb" in groups)
    return None


_OFFER_DECLINE = _compile(
    r"\b(?:don't|dont|do\s+not|haven't|havent|have\s+not)\s+(?:have|got)\b",
    r"\bno\s+(?:i\s+)?(?:don't|dont|do\s+not)\b",
    r"^(?:um\s+|well\s+)?i\s+(?:don't|dont|do\s+not)$",
)
_OFFER_OWN = _compile(
    r"\bi\s+(?:have|got|already\s+have)\s+(?:one|an?\s+(?:exercise|problem|question|example)|my\s+own|homework)\b",
    r"\bi've\s+got\s+(?:one|my\s+own|an?\s+(?:exercise|problem))\b",
    r"\bi(?:'ll|\s+will)\s+(?:write|make|draw|do|put|use)\b",
    r"\blet\s+me\s+(?:write|draw|show|put)\b",
    r"\bmy\s+own\b",
    r"^(?:yes\s+|yeah\s+|yep\s+)?i\s+do$",
    r"\bwrite\s+mine\b",
)
_OFFER_GENERATE = _compile(
    r"\bmake\s+(?:one|it|me\s+one|one\s+up|up\s+one|up\s+a\s+(?:problem|exercise|question))\b",
    r"\bgive\s+me\s+(?:one|an?\s+(?:(?:easy|easier|hard|harder|new|simple|different|random|small)\s+)?"
    r"(?:one|exercise|problem|question|example))\b",
    r"\byou\s+(?:choose|pick|decide|make\s+one|make\s+it|come\s+up)\b",
    r"\byour\s+choice\b",
    r"\bsurprise\s+me\b",
    r"\binvent\s+one\b",
    r"^(?:no|nope|not\s+really|no\s+thanks)$",
    r"^(?:yes|yeah|sure|ok|okay)\s+(?:make|give|invent|pick)\b",
)


def detect_offer_reply(text: str) -> str | None:
    """Answer to "Do you have an exercise in mind, or should I make one up?": "generate", "own" or None."""
    norm = _norm(text)
    if _any(_OFFER_DECLINE, norm):
        return "generate"
    if _any(_OFFER_OWN, norm):
        return "own"
    if _any(_OFFER_GENERATE, norm):
        return "generate"
    if is_yes(norm) or _SOFT_YES.match(norm):
        return "generate"  # "sure" (or "please") to "...or should I make one up?": make one up
    return None


# ---- both languages ----------------------------------------------------------------------------------------
# The Spanish detectors live in biobench_sar.lang and return what the English ones above return. Each *_any function
# tries the reply language's detector first and then the other one: a student may switch language mid-lesson, and
# a forced "en" setting must still understand "¿me das una pista?".

_T = TypeVar("_T")


def _spanish(name: str, text: str, empty: _T) -> _T:
    """biobench_sar.lang's Spanish detector `name` on `text`; `empty` when it is missing or fails."""
    try:
        from biobench_sar import lang
    except ImportError:
        return empty
    detector: Callable[[str], _T] | None = getattr(lang, name, None)
    if detector is None:
        return empty
    try:
        return detector(text)
    except Exception:
        log.exception("Spanish intent detector %s failed", name)
        return empty


def _norm_es(text: str) -> str:
    plain = "".join(c for c in unicodedata.normalize("NFD", (text or "").lower()) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9ñ\s]", " ", plain)).strip()


def _first(language: str, english: Callable[[], _T | None], spanish: Callable[[], _T | None]) -> _T | None:
    for detect in (spanish, english) if language == "es" else (english, spanish):
        found = detect()
        if found is not None:
            return found
    return None


def detect_mode_any(text: str, language: str = "en") -> str | None:
    """detect_mode for English or Spanish ("dame una pista", "hazme un quiz", "otro parecido")."""
    return _first(language, lambda: detect_mode(text), lambda: _spanish("detect_mode_es", text, None))


def is_quiz_stop_any(text: str) -> bool:
    """ "stop the quiz" / "termina el quiz"."""
    return is_quiz_stop(text) or bool(_spanish("is_quiz_stop_es", text, False))


_STOP_INTENT_ES = _compile(
    r"^(?:ya\s+)?(?:para|paremos|alto|basta|ya\s+basta|detente|deten|termina|terminemos|terminalo|salir|salte|"
    r"cancela|cancelalo)(?:\s+(?:ya|por\s+favor|porfa|aqui|ahi|ahora))*$",
    r"\b(?:quiero|queremos|podemos|vamos\s+a|me\s+gustaria)\s+(?:parar|terminar|salir|dejarlo|acabar|detenerlo)\b"
    r"(?!\s+(?:el|la|este|esta|mi)\s+(?:problema|ejercicio|ecuacion|paso|respuesta|tarea))",
    r"\bya\s+no\s+quiero\s+(?:seguir|continuar|jugar|hacer\s+esto)\b",
)


def is_stop_intent_any(text: str) -> bool:
    """is_stop_intent in English or Spanish: "stop", "I want to stop", "para", "quiero terminar", "termina el quiz"."""
    return is_stop_intent(text) or is_quiz_stop_any(text) or _any(_STOP_INTENT_ES, _norm_es(text))


def is_yes_any(text: str) -> bool:
    """A plain yes in English or Spanish: "yes", "sure", "ok", "sí", "claro", "dale", "de acuerdo". A lone "please",
    "fine", "bueno" or "por favor" is not: after "Do you want to end the quiz?" it keeps the quiz going."""
    return is_yes(text) or bool(_YES_ES.match(_norm_es(text)))


_REPEAT_ES = _compile(
    r"\brepit\w*",
    r"\brepetir\w*",
    r"\b(?:dilo|dimelo|leelo|lee|leemelo)\s+(?:otra\s+vez|de\s+nuevo)\b",
    r"^(?:otra\s+vez|de\s+nuevo|mande|perdon|que|como|eh)$",
    r"\bcual\s+(?:era|es)\s+(?:el|la)\s+(?:problema|pregunta|ecuacion|ejercicio)\b",
    r"\bno\s+(?:te\s+|lo\s+)?(?:escuche|oi|alcance\s+a\s+(?:oir|escuchar))\b",
)


def is_repeat_request_any(text: str) -> bool:
    """ "Can you repeat the problem?" / "¿me lo repites?" / "¿cuál era el problema?"."""
    return is_repeat_request(text) or _any(_REPEAT_ES, _norm_es(text))


# "aprender" / "enséñame" say it outright, like "learn" / "teach me"; the "¿cómo se ...?" and "explícame" forms are
# about the board when it is in use, and about the student's own problem when they name one ("¿cómo resuelvo 3x + 5?").
_LEARN_TRUSTED_ES = re.compile(r"\b(?:aprender|ensen\w*)\b")
_EXPLICIT_ES = re.compile(
    r"\d|\bes\s+igual\s+a\b|\bal\s+(?:cuadrado|cubo)\b"
    r"|\b(?:mas|menos)\s+(?:uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|x|ye)\b"
    r"|\b(?:un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+"
    r"(?:medios?|tercios?|cuartos?|quintos?|sextos?|septimos?|octavos?|novenos?|decimos?)\b"
)


_MATH_TOPIC_ES = re.compile(
    r"\b(?:fraccion|ecuacion|decimal|porcentaje|porciento|razon|proporcion|exponente|potencia|raiz|radical|algebra|"
    r"variable|expresion|desigualdad|polinomio|factor|entero|numeros?\s+negativos?|recta\s+numerica|grafica|"
    r"pendiente|funcion|angulo|area|perimetro|volumen|triangulo|circulo|rectangulo|media|mediana|promedio|"
    r"probabilidad|primo|multiplo|divisor|denominador|numerador|division|multiplicacion|suma|resta|tablas?\s+de|"
    r"jerarquia\s+de\s+operaciones|distributiva|terminos\s+semejantes|cuadratica|pitagoras|geometria|"
    r"notacion\s+cientifica|valor\s+absoluto|coordenada|lineal|sistemas?\s+de)\w*"
)
_DEIXIS_ES = re.compile(
    r"\b(?:esto|esta|este|eso|esa|ese|aqui|ahi|alla|pizarron|pizarra|mi|mis|tu|tus|la\s+siguiente|el\s+siguiente|"
    r"el\s+paso|la\s+ecuacion|el\s+problema|la\s+respuesta|el\s+resultado)\b"
)
# "¿qué es una fracción?", "no entiendo las fracciones", "ayúdame con las ecuaciones": a lesson on an empty board.
_LEARN_MORE_ES = _compile(
    r"^(?:y\s+|oye\s+|bueno\s+)?(?:que|cual)\s+(?:es|son)\s+(?P<rest>.+)",
    r"\bno\s+(?:le\s+)?entiendo\s+(?:nada\s+de\s+|bien\s+)?(?P<rest>.+)",
    r"\b(?:quiero|necesito)\s+entender\s+(?P<rest>.+)",
    r"\b(?:ayudame|me\s+ayudas|ayuda|necesito\s+ayuda)\s+(?:con|a\s+entender)\s+(?P<rest>.+)",
    r"\bhablame\s+(?:de|sobre)\s+(?P<rest>.+)",
)
# "las fracciones" -> "fracciones": the topic is said inside other sentences ("un problema fácil de {topic}",
# "Aprendiendo: {topic}"), the way English topics come without "a" or "some".
_ARTICLE_ES = re.compile(r"^(?:el|la|los|las|un|una|unos|unas)\s+")


def _norm_es_pair(text: str) -> tuple[str, str]:
    """(accent-free, accented) lowercase words of `text`, the same length, so a span found in one is the other's."""
    accented = re.sub(
        r"\s+", " ", re.sub(r"[^\w\s]|_", " ", unicodedata.normalize("NFC", (text or "").lower()))
    ).strip()
    plain = "".join(
        "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c)) or ch for ch in accented
    )
    return plain, accented


def _more_learn_es(text: str) -> str | None:
    """The topic of "¿qué es una fracción?" / "no entiendo las fracciones" / "ayúdame con las ecuaciones", accents
    kept ("fracción"); None when it is about the board or an explicit problem."""
    plain, accented = _norm_es_pair(text)
    if len(plain) != len(accented):
        accented = plain
    for pattern in _LEARN_MORE_ES:
        found = pattern.search(plain)
        if not found:
            continue
        rest = found.group("rest").strip()
        if not _MATH_TOPIC_ES.search(rest) or _DEIXIS_ES.search(rest) or _EXPLICIT_ES.search(rest):
            return None
        start = found.start("rest") + (len(found.group("rest")) - len(found.group("rest").lstrip()))
        topic = accented[start : start + len(rest)]
        words = _ARTICLE_ES.sub("", topic).split()[:MAX_TOPIC_WORDS]
        return " ".join(words) or None
    return None


def _learn_request_es(text: str, board_in_use: bool) -> str | None:
    plain = _norm_es(text)
    trusted = bool(_LEARN_TRUSTED_ES.search(plain))
    mode = _spanish("detect_mode_es", text, None)
    more = None if trusted else _more_learn_es(text)
    # "no entiendo las fracciones" / "ayúdame con las ecuaciones" can read as help mode: a named topic is a lesson.
    if not trusted and (board_in_use or (mode is not None and not (mode == "help" and more))):
        return None
    topic = _spanish("detect_learn_request_es", text, None)
    if topic is None and not trusted:
        topic = more
    if topic is None or (not trusted and _EXPLICIT_ES.search(plain)):
        return None
    return _ARTICLE_ES.sub("", topic.strip()) or None


def detect_learn_request_any(text: str, language: str = "en", *, board_in_use: bool = False) -> str | None:
    """detect_learn_request for English or Spanish ("quiero aprender fracciones", "enséñame ecuaciones")."""
    return _first(
        language,
        lambda: detect_learn_request(text, board_in_use=board_in_use),
        lambda: _learn_request_es(text, board_in_use),
    )


def detect_offer_reply_any(text: str, language: str = "en") -> str | None:
    """Answer to the lesson offer in English or Spanish: "invéntame uno" -> "generate", "tengo uno" -> "own"; a plain
    "sure" / "sí" / "claro" to "...or should I make one up?" -> "generate"."""
    found = _first(language, lambda: detect_offer_reply(text), lambda: _spanish("detect_offer_reply_es", text, None))
    if found is None and (is_yes_any(text) or _SOFT_YES_ES.match(_norm_es(text))):
        return "generate"  # a soft "bueno" / "porfa" is enough to make one up, unlike ending a quiz
    return found


class TutorSession:
    """Thread-safe: the pipeline thread and HTTP handlers both change it."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._lock = threading.RLock()
        self._rng = rng or random.Random()
        self._mode = "help"
        self._quiz: str | None = None  # None | "active" | "finished"
        self._problems: list[QuizProblem] = []
        self._invented: dict[int, QuizProblem] = {}  # index -> made-up problem, when no file has any
        self._total = 0
        self._index = 0
        self._score = 0
        self._missed = False  # a wrong answer on the current problem: it no longer counts as first-try
        self._dictated = False
        self._learn_topic: str | None = None
        self._offer_pending = False
        self._pending_stop = False  # asked "Do you want to end the quiz?": the next turn answers it
        self._turns: deque[dict[str, str]] = deque(maxlen=MAX_TURNS)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch mode. A pending lesson offer expires; practice and quiz also end the lesson."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self._lock:
            if mode != "quiz":
                self._quiz = None
                self._dictated = False
            if mode != "help":
                self._learn_topic = None
            self._offer_pending = False
            self._pending_stop = False
            self._mode = mode

    # ---- quiz ----------------------------------------------------------------
    def start_quiz(self, problems: list[QuizProblem], length: int = DEFAULT_QUIZ_LENGTH) -> None:
        """Start over. With no problems, `length` progressive problems are made up as the quiz reaches them
        (quiz_bank.invent_problem), so each one has a known text and answer like a worksheet problem."""
        with self._lock:
            self._mode = "quiz"
            self._quiz = "active"
            self._problems = list(problems)
            self._invented = {}
            self._total = len(self._problems) or max(1, length)
            self._index = self._score = 0
            self._missed = self._dictated = False
            self._learn_topic, self._offer_pending = None, False
            self._pending_stop = False

    def stop_quiz(self) -> None:
        with self._lock:
            self._quiz = None
            self._dictated = False
            self._pending_stop = False
            if self._mode == "quiz":
                self._mode = "help"

    @property
    def pending_stop(self) -> bool:
        """The student asked to stop the running quiz and was asked to confirm: the next turn answers it."""
        with self._lock:
            return self._pending_stop

    def request_stop(self) -> bool:
        """Ask to confirm stopping the running quiz; False (nothing pending) without one."""
        with self._lock:
            self._pending_stop = self._quiz == "active"
            return self._pending_stop

    def clear_stop(self) -> None:
        with self._lock:
            self._pending_stop = False

    @property
    def quiz_active(self) -> bool:
        with self._lock:
            return self._quiz == "active"

    @property
    def quiz_number(self) -> int:
        """1-based number of the current problem (the last one once the quiz is finished)."""
        with self._lock:
            return min(self._index + 1, self._total)

    @property
    def quiz_total(self) -> int:
        with self._lock:
            return self._total

    @property
    def score(self) -> int:
        """Problems solved on the first try."""
        with self._lock:
            return self._score

    def current(self) -> QuizProblem | None:
        with self._lock:
            return self._problem_at(self._index) if self._quiz == "active" else None

    def next_problem(self) -> QuizProblem | None:
        with self._lock:
            return self._problem_at(self._index + 1) if self._quiz == "active" else None

    @property
    def invented(self) -> bool:
        """The running quiz makes its problems up: no ticked file had any."""
        with self._lock:
            return self._quiz is not None and not self._problems

    def _problem_at(self, index: int) -> QuizProblem | None:
        if self._problems:
            return self._problems[index] if index < len(self._problems) else None
        if index >= self._total:
            return None
        if index not in self._invented:
            given = {p.problem for p in self._invented.values()}
            # Neighbouring numbers share a generator (only 176 one-step problems), so a draw can repeat one.
            for _ in range(20):
                problem = invent_problem(index + 1, self._total, self._rng)
                if problem.problem not in given:
                    break
            self._invented[index] = problem
        return self._invented[index]

    @property
    def dictated(self) -> bool:
        with self._lock:
            return self._dictated

    def mark_dictated(self) -> None:
        with self._lock:
            if self._quiz == "active":
                self._dictated = True

    def record(self, correct: bool) -> None:
        """A checked answer: a correct one moves to the next problem; after the last one the quiz is finished."""
        with self._lock:
            if self._quiz != "active":
                return
            if not correct:
                self._missed = True
                return
            if not self._missed:
                self._score += 1
            self._index += 1
            self._missed = self._dictated = False
            if self._index >= self._total:
                self._quiz = "finished"
                self._mode = "help"
                self._pending_stop = False

    # ---- learn ---------------------------------------------------------------
    @property
    def learn_topic(self) -> str | None:
        with self._lock:
            return self._learn_topic

    @property
    def offer_pending(self) -> bool:
        with self._lock:
            return self._offer_pending

    def start_learning(self, topic: str) -> None:
        with self._lock:
            self._learn_topic = topic
            self._offer_pending = True

    def resolve_offer(self) -> None:
        """The student answered the offer: the lesson goes on with the same topic."""
        with self._lock:
            self._offer_pending = False

    def expire_offer(self) -> None:
        """The student said something else: the offer is only answered by the very next turn."""
        with self._lock:
            if self._offer_pending:
                self._offer_pending = False
                self._learn_topic = None

    # ---- conversation memory -------------------------------------------------
    def turns(self) -> list[dict[str, str]]:
        """The last MAX_TURNS question/answer pairs, oldest first."""
        with self._lock:
            return [dict(turn) for turn in self._turns]

    def add_turn(self, q: str, a: str) -> None:
        with self._lock:
            self._turns.append({"q": (q or "")[:MAX_TURN_CHARS], "a": (a or "")[:MAX_TURN_CHARS]})

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            quiz = None
            if self._quiz is not None:
                current = self.current()
                quiz = {
                    "active": self._quiz == "active",
                    "number": self.quiz_number,
                    "total": self._total,
                    "score": self._score,
                    "problem": current.problem if current else None,
                }
            return {
                "mode": self._mode,
                "learn_topic": self._learn_topic,
                "quiz": quiz,
                "confirm_stop": self._pending_stop and self._quiz == "active",
            }
