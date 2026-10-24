"""Bilingual (en/es) helpers: which language the student is using, and the Spanish tutor intents.

The Spanish detectors return exactly what tutor.py's English ones return, so a caller can try both.
"""

from __future__ import annotations

import importlib
import logging
import re
import unicodedata
from collections.abc import Callable
from typing import Literal

log = logging.getLogger(__name__)

Language = Literal["en", "es"]
LanguageSetting = Literal["auto", "en", "es"]
LANGUAGES: tuple[Language, ...] = ("en", "es")
MAX_TOPIC_WORDS = 8
# A turn is a sentence or two; the cap keeps a pasted wall of text from tying up the query worker.
MAX_INPUT_CHARS = 2000

Detector = Callable[[str], str | None]


def _base_char(char: str) -> str:
    base = unicodedata.normalize("NFD", char)[0]
    return base if base.isascii() and base.isalpha() else char


# One character in, one character out: a span found in the plain text slices the accented text identically.
_PLAIN = {code: _base_char(chr(code)) for code in range(0xC0, 0x250) if _base_char(chr(code)) != chr(code)}


def _plain(text: str) -> str:
    return text.translate(_PLAIN)


# ---- language codes ----------------------------------------------------------
_EN_NAMES = frozenset({"en", "eng", "english", "ingles"})
_ES_NAMES = frozenset({"es", "spa", "spanish", "espanol", "castellano"})
_CODE_NAME = re.compile(r"[a-z]+")


def normalize_language(code: str | None) -> LanguageSetting:
    """The setting for a code or locale: "es-MX" / "Español (México)" -> "es", "English" -> "en", else "auto"."""
    found = _CODE_NAME.search(_plain((code or "")[:64].lower()))
    key = found.group() if found else ""
    if key in _EN_NAMES:
        return "en"
    if key in _ES_NAMES:
        return "es"
    return "auto"


# ---- language detection ------------------------------------------------------
def _word_set(words: str) -> frozenset[str]:
    return frozenset(words.split())


# Accent-free, and never in both sets: words shared by the two languages ("no", "me", "a", "quiz", "total")
# or that are math in English ("sin", "tan", "dice") vote for neither.
_ES_WORDS = _word_set(
    """
    el la los las un una unos unas lo le les se te mi mis tu tus su sus nos yo usted ustedes ella ellos nosotros
    este esta esto estos estas ese esa eso esos esas aquel de del al en y o que ni pero porque aunque como cuando
    donde cual cuales quien cuanto cuanta cuantos cuantas por para con entre sobre hasta desde hacia segun tambien
    entonces luego despues antes ahora ahorita ya aqui aca ahi alli alla muy mas menos poco mucho mucha muchos
    tanto todo toda todos todas nada algo alguno alguna ninguno ninguna otro otra otros otras mismo misma cada si
    pues bueno bien mal asi siguiente proximo parecido parecida
    es estoy estamos estan soy eres somos ser estar hay tengo tienes tiene tenemos tienen tener puedo puedes puede
    podemos pueden poder quiero quieres quiere queremos quisiera necesito necesitas necesita sabes sabe saber
    entiendo entiendes entiende entender entendi comprendo sigue sigo seguir voy vas va vamos ir hago haces hace
    hacemos hacen hacer haz hazme dame ayuda ayudame ayudas ayudar ayudarme explica explicame explicar ensename
    ensenar aprender resolver resuelve resuelvo sumar suma sumo restar resta multiplicar multiplica dividir
    despejar despeja calcular calcula escribir escribe escribo mira veo ves termina terminar termine acabe falta
    faltan sirve significa queda pasa doy das pon ponme inventame examiname escoge elige creo parece atorado
    atorada ando ver espera esperame ocupo agarro practicar practicando preguntame cualquiera tuyo mio mia mios
    propio
    nel nop sip simon neta chale aja porfis plis nomas onda tantito orita xq dale mames practiquemos atorao
    paso pasos problema problemas ecuacion ecuaciones fraccion fracciones ejercicio ejercicios respuesta
    respuestas pregunta preguntas numero numeros resultado raiz cuadrado cubo tarea examen prueba pizarron pizarra
    hoja ejemplo tema clase lado parte igual mitad medio tercio tercios cuarto cuartos vez veces punto signo valor
    letra pista practica porcentaje ciento grafica linea recta angulo triangulo perimetro duda maestra maestro profe
    evaluacion
    cero uno dos tres cuatro cinco seis siete ocho nueve diez doce trece catorce quince veinte treinta cuarenta
    cincuenta cien mil
    hola gracias oye mande porfa vale orale andale chido listo lista claro adios
    """
)
_EN_WORDS = _word_set(
    """
    the an and or but if so of to in on at by for from with about into over under up out off as is are was were
    be been being am i i'm im i've i'll i'd you you're your we they it it's its this that that's these those there
    here my mine she him his her them their our what what's whats how how's why where when which who whose do does
    did done don't dont doesn't didn't can can't cant cannot could couldn't would should will won't isn't aren't
    shall may might must have haven't had get got give make show tell let let's lets help need want wants know
    think see look go try put take find keep check wait stop start finish quit end please thanks thank hello hi
    hey yes yeah yep nope right wrong sure not just also too very really again now then next step steps stuck
    hint solve solving answer question problem problems equation equations fraction fractions number numbers
    exercise practice test exam learn teach explain understand mean add subtract multiply divide plus minus times
    equals equal divided square root half third quarter another more one two three four five six seven eight nine
    eleven twelve twenty hundred thousand zero like same other else much many some any all every each first last
    write draw board back way side value whole because neither either
    """
)
_SPANISH_PUNCT = re.compile(r"[¿¡]")
_SPANISH_LETTERS = re.compile(r"[áéíóúüñ]")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_MATH_CHARS = frozenset("0123456789=+-−*/^()[]{}<>≤≥≠×÷·√%.")


def _neighbor(text: str, index: int, step: int) -> str:
    """The first non-space character from `index` walking by `step`, or "" past either end. Never copies `text`."""
    while 0 <= index < len(text) and text[index].isspace():
        index += step
    return text[index] if 0 <= index < len(text) else ""


def _is_variable(text: str, start: int, end: int) -> bool:
    """A lone letter beside numbers or operators ("y = 2x", "3 y 4"), or on its own, is math rather than a word."""
    before = _neighbor(text, start - 1, -1)
    after = _neighbor(text, end, 1)
    return (not before and not after) or before in _MATH_CHARS or after in _MATH_CHARS


def _vote(word: str, previous: str | None, previous_vote: str | None) -> str | None:
    if _plain(word) in _ES_WORDS:
        return "es"
    if word in _EN_WORDS:
        return "en"
    if word == "me":  # "help me" / "quiz me"; in "¿me ayudas?" the next word decides
        if previous == "no":  # "no me sale" / "no me acuerdo": English says "not me"
            return "es"
        return "en" if previous == "quiz" else previous_vote
    return None


def detect_language(text: str, default: str = "en") -> Language:
    """The language ("en" / "es") the student said or typed; `default` when nothing leans either way ("ok", math)."""
    fallback: Language = "es" if normalize_language(default) == "es" else "en"
    capped = (text or "")[:MAX_INPUT_CHARS]
    lowered = unicodedata.normalize("NFC", capped.lower()).replace("’", "'").replace("‘", "'")
    spanish = 2 * len(_SPANISH_PUNCT.findall(lowered))
    english = 0
    previous: str | None = None
    previous_vote: str | None = None
    for found in _WORD.finditer(lowered):
        word = found.group()
        if len(word) == 1 and _is_variable(lowered, found.start(), found.end()):
            previous = previous_vote = None
            continue
        vote = _vote(word, previous, previous_vote)
        if _SPANISH_LETTERS.search(word):
            spanish += 1
        if vote == "es":
            spanish += 1
        elif vote == "en":
            english += 1
        previous, previous_vote = word, vote
    if spanish > english:
        return "es"
    if english > spanish:
        return "en"
    return fallback


# ---- Spanish intents: shared text handling -------------------------------------
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[,.;:!?¿¡…]")


def _normalize(text: str) -> tuple[str, str]:
    """(plain, accented): lowercase words, punctuation folded to "," so patterns stop at clause breaks.

    Both strings have the same length, so a span matched in the plain one also slices the accented one.
    """
    capped = (text or "")[:MAX_INPUT_CHARS]
    lowered = unicodedata.normalize("NFC", capped.lower()).replace("’", "'").replace("‘", "'")
    tokens: list[str] = []
    for token in _TOKEN.findall(lowered):
        if token[0].isalnum():
            tokens.append(token)
        elif tokens and tokens[-1] != ",":
            tokens.append(",")
    if tokens and tokens[-1] == ",":
        tokens.pop()
    accented = " ".join(tokens)
    return _plain(accented), accented


def _flat(plain: str) -> str:
    return " ".join(token for token in plain.split() if token != ",")


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


def _any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


# Fillers that may surround a one-phrase request: "ok, otro quiz porfa" / "va, más ejercicios".
_FILLER_ES = (
    r"(?:ya|oye|ok|okay|va|vale|sale|dale|bueno|pues|ahora|ahorita|entonces|porfa|porfis|plis|por\s+favor|profe|"
    r"maestra|maestro|andale|orale|y|mejor|a\s+ver)"
)


def _alone(pattern: re.Pattern[str], plain: str) -> bool:
    """`pattern` is a whole clause ("otro quiz" / "ok, otro quiz"), not part of one ("el otro quiz estuvo fácil")."""
    return any(pattern.fullmatch(clause) for clause in plain.split(" , "))


# What comes right before a command that makes it not one: negated ("no quiero un quiz" / "no quiero que me
# hagas un examen"), asked about ("¿cuándo termina el quiz?") or a purpose clause ("falta poco para terminar el quiz").
_NOT_A_COMMAND = re.compile(
    r"(?:\b(?:no|nunca|ni|tampoco)\s+"
    r"(?:(?:quiero|queria|quisiera|queremos|puedo|podemos|puedes|necesito|debo|debemos|voy\s+a|vamos\s+a|"
    r"tengo\s+que|hay\s+que|me\s+gusta|me\s+gustaria)\s+(?:que\s+)?)?"
    r"|\b(?:cuando|como|donde|cuanto|cuanta|cuantos|cuantas|cual|cuales|quien|porque|por\s+que|para\s+que|"
    r"que\s+hora|para|antes\s+de|despues\s+de|hasta|sin|al|se)\s+)$"
)
# "¿por qué no me haces un quiz?" suggests one; "por que no quiero hacer un quiz" (no accent: "porque") refuses it.
_SUGGESTION_ES = re.compile(r"\bpor\s+que\s+no\s+$")
_WANT_ES = re.compile(r"(?:quiero|queremos|quisiera|queria|necesito|ocupo|puedo|podemos|debo|me\s+gusta)\b")
# "después me haces un quiz" puts it off: not a mode change now. "luego luego" is Mexican for "right away".
_LATER_ES = re.compile(r"\b(?:despues|(?<!\bluego )luego|al\s+rato|mas\s+tarde|otro\s+dia)\s+$")
# Longer than any _NOT_A_COMMAND prefix, so looking back no further than this gives the same answer, in linear time.
_LOOKBACK = 48


def _negated(plain: str, start: int) -> bool:
    window = max(0, start - _LOOKBACK)
    if not _NOT_A_COMMAND.search(plain, window, start):
        return False
    return not (_SUGGESTION_ES.search(plain, window, start) and not _WANT_ES.match(plain, start))


def _commanded(patterns: tuple[re.Pattern[str], ...], plain: str) -> bool:
    return any(not _negated(plain, m.start()) for p in patterns for m in p.finditer(plain))


def _requested(patterns: tuple[re.Pattern[str], ...], plain: str, unless: re.Pattern[str] | None = None) -> bool:
    """Commanded now: not negated, not put off ("luego"), and not right after `unless` ("hay otro así")."""
    for pattern in patterns:
        for found in pattern.finditer(plain):
            start = found.start()
            window = max(0, start - _LOOKBACK)
            if _negated(plain, start) or _LATER_ES.search(plain, window, start):
                continue
            if unless is None or not unless.search(plain, window, start):
                return True
    return False


# ---- Spanish intents: mode -----------------------------------------------------
_QUIZ_WORD = r"(?:quiz|quizz|quizzito|quizito|prueba|pruebita|examen|examencito|evaluacion|evaluacioncita|cuestionario)"
# "prueba" and "evaluación" are feminine: "termina la prueba", "para esta evaluación", "párale a la prueba".
_ARTICLE = r"(?:el|la|este|esta|ese|esa|mi|del)"
_THE_QUIZ = rf"(?:ya\s+)?(?:(?:de|con|a|al)\s+)?(?:{_ARTICLE}\s+)?{_QUIZ_WORD}\b"
_CLOSING_FILLER_ES = (
    r"(?:ya|porfa|porfis|plis|pls|please|por\s+favor|gracias|profe|maestra|maestro|ahora|ahorita|pues|de\s+una\s+vez)"
)
# In Mexican math class "hacer la prueba" is also checking an answer: "hagamos la prueba con x = 2", "la prueba
# del nueve", and a bare "¿hacemos la prueba?" right after solving. With "haz / hagamos / hacemos / vamos a hacer"
# it is a quiz only when a topic follows ("hagamos la prueba de fracciones"). "hazme / dame / empecemos la prueba"
# are quizzes.
_CHECKING = (
    r"\s+(?:con|a|al|del|de\s+(?:la|el|esta|este|esa|ese|tu|mi)|sustituyendo|para\s+(?:ver|comprobar|checar|saber))\b"
)
_CHECK_VERBS = ("haz", "hagamos", "hacemos", "vamos a hacer")
_AFTER_CHECK_VERB = "|".join(rf"(?<=\b{verb} la )" for verb in _CHECK_VERBS)
_PRUEBA = (
    rf"(?!(?<=\bla )prueb(?:a|ita){_CHECKING})"
    rf"(?!(?:{_AFTER_CHECK_VERB})prueb(?:a|ita)(?!\s+de\s))"
    r"prueb(?:a|ita)"
)
_QUIZ_ASKED = rf"(?:quiz|quizz|quizzito|quizito|{_PRUEBA}|examen|examencito|evaluacion|evaluacioncita|cuestionario)\b"
_A_QUIZ = rf"(?:un|una|el|la|este|esta|otro|otra)\s+(?:\w+\s+)?{_QUIZ_ASKED}"

# A bare "si" opens only before a comma: "sí, termina el quiz"; "si termina el quiz ..." is "if the quiz ends".
_OPENING_FILLER_ES = (
    r"(?:ya|oye|mejor|bueno|ok|okay|okey|ahora|pues|porfa|porfis|plis|por\s+favor|profe|maestra|maestro|no|va|sale|"
    r"dale|orale|andale|sip|entonces|ay|y\s+ya|a\s+ver|nomas|si\s+,)"
)
_WHEN_ES = r"(?:manana|hoy|ayer|lunes|martes|miercoles|jueves|viernes|sabado|domingo|el|la|los|las|esta|este)"
_DAY_ES = (
    r"(?:manana|hoy|ayer|antier|lunes|martes|miercoles|jueves|viernes|sabado|domingo|semana|proxim[oa]|tarde|"
    r"noche|temprano|pronto)"
)
# "termina el quiz de mañana a las tres" says when it ends; it is not a command.
_NOT_NOW_ES = (
    rf"\s+(?:,\s+)?(?:a\s+las?|a\s+la\s+una|en\s+\w+\s+minutos?|cuando|hasta|de\s+{_WHEN_ES}|"
    rf"(?:el|la|este|esta|en\s+la)\s+{_DAY_ES}|{_DAY_ES})\b"
)
# Why a student quits, said right after a bare "para el quiz": "ya me cansé", "que ya no quiero", "ya entendí".
# Without "ya", "para el examen no quiero sacar seis" / "terminé el tema 3" / "me voy a preparar" are "for the exam".
_QUIT_REASON_ES = (
    r"(?:(?:porque|que|pues|y|es\s+que)\s+)?(?:ya\s+(?:no\s+(?:quiero|queremos|aguanto|puedo)|estuvo|entendi|"
    r"le\s+entendi|termine|acabe)|no\s+(?:quiero|queremos)(?=\s*(?:,|$)|\s+(?:seguir|continuar|mas)\b)|"
    r"no\s+aguanto|no\s+puedo\s+mas|(?:ya\s+)?me\s+(?:canse|aburri|harte|fastidie|dio\s+sueno)|"
    r"(?:ya\s+)?(?:estoy|ando)\s+(?:muy\s+)?(?:cansad|aburrid|hart)[oa]|tengo\s+sueno|"
    r"(?:ya\s+)?me\s+voy(?=\s*(?:,|$))|basta|(?:ya\s+)?fue\s+suficiente)\b"
)
# "para" / "termina" / "acaba" are also "for" / "it ends": only commands when they open the sentence. A bare
# "para el quiz" may then only be followed by fillers or a reason to quit, since "para el examen, ¿qué temas
# vienen?" is "for the exam"; "termina el quiz ..." may say anything except when it ends.
_STOP_OPENING_ES = re.compile(
    rf"^(?:{_OPENING_FILLER_ES}\s+(?:,\s+)?)*(?:"
    rf"(?:para\s+(?:,\s+)?)?para\s+{_THE_QUIZ}(?:\s+(?:,\s+)?(?:{_CLOSING_FILLER_ES}|y\s+ya))*"
    rf"(?:$|\s+(?:,\s+)?{_QUIT_REASON_ES})"
    rf"|(?:parale|parele|termina|terminale|acaba|acabale|acaba\s+con|termina\s+con)\s+{_THE_QUIZ}(?!{_NOT_NOW_ES}))"
)
# "¿cómo salgo del examen?" / "¿cómo le hago para salir del quiz?" ask to quit. Not "terminar": "¿cómo le hago
# para terminar el quiz?" may be asking how to finish it.
_HOW_TO_QUIT_ES = re.compile(
    r"\bcomo\s+(?:le\s+)?(?:(?:hago|hacemos|puedo|podemos|se\s+puede)\s+(?:para\s+)?)?"
    rf"(?:salgo|salimos|me\s+salgo|salir|salirme|salirnos|parar|cancelar|dejar|quitar|cerrar)\s+{_THE_QUIZ}"
)
_STOP_ES = _compile(
    rf"\b(?:terminemos|paremos|detengamos|cancelemos|salgamos|acabemos|dejemos|finalicemos|cerremos|quitemos)\s+{_THE_QUIZ}",
    rf"\b(?:terminar|parar|detener|cancelar|salir|salirme|acabar|dejar|finalizar|cerrar|quitar)\s+{_THE_QUIZ}",
    rf"\b(?:cancela|deten|detente|quita|cierra|finaliza|sal|salte|sacame|sacanos|me\s+salgo)\s+{_THE_QUIZ}",
    # "mejor dejamos el examen para mañana"; a bare "dejamos el examen en la mesa" says where it was left.
    rf"\b(?:mejor|ya|ahi)\s+dejamos\s+{_THE_QUIZ}",
    rf"\bdejamos\s+{_THE_QUIZ}\s+(?:para\s+(?:otro|otra|despues|luego|manana|el|la)|ahi|aqui|hasta\s+aqui|por\s+hoy)\b",
    rf"\bya\s+no\s+(?:quiero|queremos|quiere)\s+(?:(?:hacer|seguir|con|{_ARTICLE}|un|una|otro|otra|mas)\s+)*"
    rf"{_QUIZ_WORD}\b",
    rf"\bno\s+(?:quiero|queremos)\s+(?:seguir|continuar)\s+(?:con\s+)?{_THE_QUIZ}",
    r"\bno\s+mas\s+(?:quiz(?:z|zes|es)?|examen(?:es)?|pruebas?|evaluacion(?:es)?|cuestionarios?)\b",
    r"\b(?:ya\s+)?no\s+(?:quiero\s+|queremos\s+)?mas\s+preguntas\b",
    rf"\b(?:basta|suficiente|ya\s+estuvo|(?:ya\s+)?me\s+(?:canse|aburri|harte))\s+"
    rf"(?:(?:de|con)\s+(?:{_ARTICLE}\s+)?|del\s+){_QUIZ_WORD}\b",
    # Not "es suficiente prueba de que ...": "enough proof that ...".
    rf"\bsuficiente\s+{_QUIZ_WORD}\b(?!\s+(?:de|para)\b)",
    rf"\bme\s+(?:aburrio|canso|harto|fastidio)\s+{_THE_QUIZ}(?!{_NOT_NOW_ES})",
)
_QUIZ_ES = _compile(
    r"\b(?:examiname|evaluame|examinanos|evaluanos|preguntame|preguntanos|tomame\s+la\s+leccion)\b",
    r"\bponme\s+a\s+prueba\b",
    r"\b(?:me|nos)\s+(?:tomas|tomarias|puedes\s+tomar|podrias\s+tomar)\s+la\s+leccion\b",
    r"\b(?:quiero|queremos|quisiera|me\s+gustaria|necesito)\s+que\s+(?:me|nos)\s+(?:preguntes|hagas\s+preguntas|"
    r"examines|evalues|tomes\s+la\s+leccion)\b",
    r"\b(?:hazme|haznos|dame|danos|ponme|ponnos|aplicame|preparame|haz|pon|prepara|mandame|echame|pasame|avientame|"
    r"aventame)\s+" + _A_QUIZ,
    r"\b(?:me|nos)\s+(?:haces|hagas|harias|puedes\s+hacer|podrias\s+hacer|pones|podrias\s+poner|das|aplicas|"
    r"mandas)\s+" + _A_QUIZ,
    r"\b(?:quiero|queremos|quisiera|queria|me\s+gustaria|necesito|ocupo)\s+(?:hacer\s+|tomar\s+|presentar\s+|"
    r"tener\s+)?" + _A_QUIZ,
    r"\b(?:podemos|puedes|podrias|podriamos)\s+(?:hacer|aplicar|poner)(?:me|nos)?\s+" + _A_QUIZ,
    rf"\bhay\s+que\s+hacer\s+(?:un|una|otro|otra)\s+(?:\w+\s+)?{_QUIZ_ASKED}",
    r"\b(?:empecemos|empieza|empezar|comienza|comenzar|comencemos|inicia|iniciar|iniciemos|hagamos|hacemos|"
    rf"arranca|arranquemos|vamos\s+a\s+hacer)\s+(?:con\s+)?(?:(?:un|una|el|la|este|esta|otro|otra)\s+)?{_QUIZ_ASKED}",
    rf"\b(?:vamos|vamonos|pasemos|pasame|pasanos|quiero\s+pasar|quiero\s+ir)\s+(?:al|a\s+(?:la|una?))\s+{_QUIZ_ASKED}",
    rf"\bmodo\s+(?:de\s+)?{_QUIZ_WORD}\b",
    # "pregúntame" is "test me"; "tengo una pregunta" / "te hago una pregunta" are the student asking.
    r"\b(?:hazme|haznos|ponme|dame|me\s+haces|nos\s+haces|me\s+puedes\s+hacer|puedes\s+hacerme)\s+"
    r"(?:(?:una|unas|algunas|otra|otras|mas|dos|tres|cuatro|cinco|diez)\s+)?pregunt(?:a|as|ita|itas)\b",
)
_HARDER = (
    r"mas\s+(?:dificil|facil|complicad[oa]|retador[a]?|sencill[oa]|dificilito|dificilita|facilito|facilita|"
    r"pesad[oa]|corto|largo|rapido)"
)
_QUIZ_ALONE_ES = re.compile(
    rf"(?:{_FILLER_ES}\s+)*(?:otro|otra|un|una)\s+{_QUIZ_ASKED}"
    rf"(?:\s+(?:pero\s+)?(?:{_HARDER}|mas|rapido|rapidito|cortito|facil|dificil|de\s+\w+))?(?:\s+{_FILLER_ES})*"
)
_LIKE_THIS = r"(?:parecid[oa]s?|similar(?:es)?|asi|como\s+(?:este|esta|estos|estas|ese|esa|esto|eso|el\s+anterior|la\s+anterior))"
# Which earlier item "igual a la ..." points at: "la anterior", "la tres" (worksheet item 3), "la que hicimos".
_ITEM_REF = (
    r"(?:anterior(?:es)?|de\s+(?:antes|arriba|abajo|atras)|que|primer[oa]|segund[oa]|tercer[oa]|cuart[oa]|"
    r"quint[oa]|ultim[oa]|otr[oa]|\d+|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b"
)
# "otro igual a este" / "otra igual a la anterior" ask for one; "otro igual a cinco" / "a la mitad" is dictation.
_SAME = rf"(?:igual(?!\s+a\s+(?!(?:este|esta|ese|esa|esto|eso)\b|(?:la|las|los|lo)\s+{_ITEM_REF}))|igualit[oa])"
_EXERCISE = r"(?:problema|ejercicio|ejemplo|pregunta|ecuacion|fraccion|operacion)"
# Not "pregunta" ("otra pregunta" mid-quiz is the next question) nor "ejemplo" (a lesson asking for one more).
_PRACTICE_ITEM = r"(?:ejercicio|problema|ecuacion|operacion)"
_GIVE_ME = (
    r"(?:dame|danos|ponme|ponnos|pon|hazme|mandame|pasame|echame|avientame|quiero|queremos|necesito|ocupo|"
    r"quisiera|me\s+(?:das|pones|pasas|mandas))"
)
# "hay otro problema así en la hoja" / "es otra fracción igual" describe the worksheet; they ask for nothing.
_STATEMENT_ES = re.compile(r"\b(?:hay|habia|es|son|era|eran|fue|tengo)\s+$")
_PRACTICE_ES = _compile(
    # "uno igual" is left out: "dos más uno igual a tres" is dictation, not a request.
    rf"\b(?:otro|otra)\s+(?:mas\s+)?(?:{_EXERCISE}\s+)?(?:mas\s+)?(?:{_LIKE_THIS}|{_SAME})\b",
    rf"\b(?:uno|una)\s+(?:mas\s+)?{_LIKE_THIS}\b",
    rf"\b(?:un|una)\s+{_EXERCISE}\s+{_LIKE_THIS}\b",
    rf"\b{_EXERCISE}s\s+(?:mas\s+)?{_LIKE_THIS}\b",
    r"\bmas\s+practica\b",
    r"\b(?:practicar|practiquemos)\s+(?:mas|otro|otra|un\s+poco)\b",
    r"\bquiero\s+practicar\b",
    rf"\b(?:practiquemos|vamos\s+a\s+practicar)\b(?!{_NOT_NOW_ES})",
    r"\b(?:seguir|sigamos|sigo|continuar|continuemos)\s+practicando\b",
    r"\bmodo\s+(?:de\s+)?practica\b",
    r"\b(?:problemas|ejercicios)\s+de\s+practica\b",
    rf"\b{_GIVE_ME}\s+(?:hacer\s+|resolver\s+)?(?:otro|otra|otros|otras|mas)\s+{_PRACTICE_ITEM}s?\b",
    rf"\b{_GIVE_ME}\s+(?:otro|otra|uno|una)\s+(?:{_PRACTICE_ITEM}\s+)?{_HARDER}\b",
    r"\b(?:dame|danos|ponme|ponnos|pasame|mandame|echame|avientame)\s+(?:(?:unos|unas|algunos|algunas|varios|"
    r"varias)\s+)?(?:ejercicios|problemas|ecuaciones|operaciones)\b",
    # Not "sigamos con otro tema" / "otra" alone ("otra" mid-quiz is the next question).
    rf"\b(?:sigamos|seguimos|continuemos|vamos)\s+con\s+(?:otro(?:\s+{_PRACTICE_ITEM})?|otra\s+(?:ecuacion|operacion))"
    r"(?=\s*(?:,|$)|\s+(?:porfa|porfis|plis|por\s+favor|ya|igual|parecid[oa]|similar|asi)\b)",
)
# "hay más ejercicios en la hoja" is about the worksheet, so a bare "más ejercicios" must be the whole request.
_PRACTICE_ALONE_ES = re.compile(
    rf"(?:{_FILLER_ES}\s+)*(?:(?:otro|otra)\s+{_PRACTICE_ITEM}(?:\s+mas)?(?:\s+de\s+\w+)?|mas\s+{_PRACTICE_ITEM}s"
    rf"(?:\s+de\s+\w+)?|(?:otro|otra|uno|una)\s+(?:{_PRACTICE_ITEM}\s+)?{_HARDER}|(?:hay\s+que\s+)?"
    r"practicar(?:\s+(?:mas|mucho|un\s+poco(?:\s+mas)?|otro\s+rato))?)"
    rf"(?:\s+{_FILLER_ES})*"
)
_HELP_ES = _compile(
    # "yo puedo sin ayuda" turns it down.
    r"\b(?<!\bsin )ayud(?:a|ame|arme|as|en|enme|ita|itas|adita|aditas|eme)\b",
    r"\bauxilio\b",
    r"\b(?:siguiente|proximo)\s+paso\b",
    # "q" / "k" are how students type "qué": "q hago", "k sigue".
    r"\b(?:que|q|k)\s+(?:sigue|hago|debo\s+hacer|tengo\s+que\s+hacer|hay\s+que\s+hacer|mas\s+hago|le\s+hago)\b",
    r"\bcomo\s+le\s+(?:hago|hacemos)\b",
    r"\by\s+ahora\s+(?:que|como)\b",
    r"\bno\s+(?:le\s+|lo\s+|la\s+)?(?:entiendo|entendi|comprendo|comprendi|capto|cacho|agarro|hallo)\b",
    r"\bno\s+(?:le\s+|lo\s+|la\s+)?estoy\s+(?:entendiendo|comprendiendo|captando|cachando)\b",
    r"\bsigo\s+sin\s+(?:entender|comprender|saber)\b",
    r"\bno\s+le\s+se\b",
    r"\bno\s+me\s+(?:sale|salen|salio|queda\s+claro|acuerdo)\b",
    r"\bno\s+se\s+(?:como|que\s+hacer|que\s+sigue|que\s+hago|por\s+donde)\b",
    # "no sé hacerlo" / "no sé resolverlo"; "no se hace así" (it isn't done that way) has no infinitive.
    r"\bno\s+se\s+(?:hacer|resolver|despejar|sacar)\w*",
    # "¿cómo se hace?" on its own; "¿cómo se hace una gráfica?" is a lesson request.
    # Not "¡así es como se hace!", which is the student showing off.
    r"(?<!\bes )(?<!\basi )\bcomo\s+se\s+hace(?:\s+(?:esto|eso|este|esta|ese|esa|aqui))?(?=\s*(?:,|$))",
    r"\bme\s+(?:atore|atasque|trabe|perdi|confundi|revolvi|bloque\w*)\b",
    r"\b(?:estoy|toy|ando|me\s+quede|quede)\s+en\s+blanco\b",
    r"\b(?:h?echame|h?echanos|me\s+h?echas|nos\s+h?echas)\s+(?:la|una)\s+mano(?:ita)?\b",
    r"\bpista\b",
)
_HELP_CLAUSE_ES = re.compile(
    rf"(?:{_FILLER_ES}\s+)*(?:me\s+(?:explicas|explica|explicarias|puedes\s+explicar|podrias\s+explicar)|explicame)"
    r"(?:\s+(?:otra\s+vez|de\s+nuevo|porfa|porfis|plis|por\s+favor|profe|maestra|maestro))*"
)
_VERY_ES = r"(?:(?:bien|muy|super|todo|bastante|medio|tan|algo|un\s+poco|un\s+poquito)\s+)?"
# Whisper may write the spoken "atorao".
_STUCK_WORD = r"(?:(?:atorad|atascad|trabad|confundid)[oa]s?|atorao|atascao|trabao)"
_STUCK_ES = re.compile(
    rf"\b(?:estoy|toy|stoy|ando|andamos|estamos|me\s+quede|me\s+siento|sigo|quede)\s+{_VERY_ES}"
    rf"(?:{_STUCK_WORD}|perdid[oa]s?|stuck)\b"
)
# A bare "atorado" is stuck, like English "stuck", but only as the whole clause: "el problema no está atorado"
# and "la variable está confundida con y" are about something else. A bare "perdido" is not ("el número perdido").
_STUCK_ALONE_ES = re.compile(
    rf"(?:{_FILLER_ES}\s+)*{_VERY_ES}{_STUCK_WORD}(?:\s+(?:con|en)\s+.*|\s+(?:aqui|aca|otra\s+vez|de\s+nuevo))?"
    rf"(?:\s+{_FILLER_ES})*"
)
_NOT_STUCK_ES = re.compile(r"\b(?:no|nunca|tampoco)\s+$")
_HELP_ALONE_ES = re.compile(r"^y\s+(?:ahora|luego|despues|entonces)(?:\s+que)?$")
_NO_HELP_ES = re.compile(r"\bno\s+(?:necesito|ocupo|quiero)\s+(?:tu\s+|mas\s+|nada\s+de\s+)?ayuda\b")


def _stuck(plain: str) -> bool:
    """ "ando atorado" / "atorada", but not "ya no estoy atorado" nor "no estoy tan atorado"."""
    return _alone(_STUCK_ALONE_ES, plain) or any(
        not _NOT_STUCK_ES.search(plain, max(0, m.start() - _LOOKBACK), m.start()) for m in _STUCK_ES.finditer(plain)
    )


def _is_quiz_stop(plain: str) -> bool:
    return bool(_STOP_OPENING_ES.search(plain)) or bool(_HOW_TO_QUIT_ES.search(plain)) or _commanded(_STOP_ES, plain)


def is_quiz_stop_es(text: str) -> bool:
    """An explicit "termina el quiz": "ayuda" / "estoy atorado" during a quiz asks for a hint, not to quit."""
    return _is_quiz_stop(_normalize(text)[0])


def detect_mode_es(text: str) -> str | None:
    """The mode ("help" | "practice" | "quiz") a student's Spanish asks for, or None to stay in the current one."""
    plain = _normalize(text)[0]
    if _is_quiz_stop(plain):
        return "help"
    if _requested(_QUIZ_ES, plain) or _alone(_QUIZ_ALONE_ES, plain):
        return "quiz"
    if _requested(_PRACTICE_ES, plain, unless=_STATEMENT_ES) or _alone(_PRACTICE_ALONE_ES, plain):
        return "practice"
    if _NO_HELP_ES.search(plain):
        return None
    if _any(_HELP_ES, plain) or _stuck(plain) or _alone(_HELP_CLAUSE_ES, plain) or _HELP_ALONE_ES.search(_flat(plain)):
        return "help"
    return None


# ---- Spanish intents: "quiero aprender ..." -------------------------------------
# "aprender" / "enséñame" say what they want outright; "explícame" and the "¿cómo se ...?" forms must also name
# a math topic, or "¿cómo resuelvo esto?" would turn a question about the board into a lesson.
_LEARN_TRUSTED_ES = _compile(
    r"\baprender(?:\s+(?:sobre|acerca\s+de))?\s+(?P<rest>[^,]+)",
    r"\b(?:ensename|ensenanos|ensenarme|me\s+(?:puedes|podrias|quieres|puede|podria)\s+ensenar|me\s+ensenas|"
    r"me\s+ensenarias|me\s+ensena)(?:\s+(?:sobre|acerca\s+de))?\s+(?P<rest>[^,]+)",
)
_LEARN_EXPLAIN_ES = re.compile(
    r"\b(?:explicame|explicanos|explicarme|me\s+(?:puedes|podrias|puede|podria)\s+explicar|me\s+explicas|"
    r"me\s+explica|muestrame|mostrarme|me\s+(?:puedes|podrias)\s+mostrar|me\s+muestras)"
    r"(?:\s+(?:sobre|acerca\s+de))?\s+(?P<rest>[^,]+)"
)
_HOWTO_ES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "infinitive",
        re.compile(r"\bcomo\s+(?:se\s+)?(?:le\s+)?(?:hago|hace|hacemos|haces|hacen|hacer)\s+para\s+(?P<phrase>[^,]+)"),
    ),
    (
        "infinitive",
        re.compile(
            r"\bcomo\s+(?:se\s+)?(?:le\s+)?(?:puedo|podemos|puede|puedes|podria|podriamos|debo|debemos|debe|"
            r"tengo\s+que|tenemos\s+que|tiene\s+que|hay\s+que)\s+(?P<phrase>[^,]+)"
        ),
    ),
    # "¿cómo se le saca la raíz a un número?" / "¿cómo le saco la raíz?"
    ("impersonal", re.compile(r"\bcomo\s+se\s+(?:les?\s+)?(?P<phrase>(?P<verb>[a-z]+)\s+[^,]+)")),
    ("personal", re.compile(r"\bcomo\s+(?:les?\s+)?(?P<phrase>(?P<verb>[a-z]+)\s+[^,]+)")),
)
_VERB_FORMS = {
    "sumar": "suma suman sumo sumas sumamos",
    "restar": "resta restan resto restas",
    "multiplicar": "multiplica multiplican multiplico multiplicas",
    "dividir": "divide dividen divido divides",
    "resolver": "resuelve resuelven resuelvo resuelves resolvemos",
    "despejar": "despeja despejan despejo despejas",
    "simplificar": "simplifica simplifican simplifico simplificas",
    "calcular": "calcula calculan calculo calculas",
    "sacar": "saca sacan saco sacas",
    "encontrar": "encuentra encuentran encuentro encuentras",
    "hacer": "hace hacen hago haces",
    "graficar": "grafica grafican grafico graficas",
    "factorizar": "factoriza factorizan factorizo factorizas",
    "convertir": "convierte convierten convierto conviertes",
    "reducir": "reduce reducen reduzco reduces",
    "elevar": "eleva elevan elevo elevas",
    "obtener": "obtiene obtienen obtengo obtienes",
    "medir": "mide miden mido mides",
    "comparar": "compara comparan comparo comparas",
    "redondear": "redondea redondean redondeo redondeas",
    "ordenar": "ordena ordenan ordeno ordenas",
    "usar": "usa usan uso usas",
    "aplicar": "aplica aplican aplico aplicas",
    "escribir": "escribe escriben escribo escribes",
    "leer": "lee leen leo lees",
    "representar": "representa representan represento representas",
    "dibujar": "dibuja dibujan dibujo dibujas",
    "trazar": "traza trazan trazo trazas",
    "contar": "cuenta cuentan cuento cuentas",
    "comprobar": "comprueba comprueban compruebo compruebas",
    "distribuir": "distribuye distribuyen distribuyo distribuyes",
    "sustituir": "sustituye sustituyen sustituyo sustituyes",
    "combinar": "combina combinan combino combinas",
    "agrupar": "agrupa agrupan agrupo agrupas",
    "pasar": "pasa pasan paso pasas",
    "quitar": "quita quitan quito quitas",
    "entender": "entiende entienden entiendo entiendes",
    "estimar": "estima estiman estimo estimas",
    "promediar": "promedia promedian promedio promedias",
}
_INFINITIVES = {form: verb for verb, forms in _VERB_FORMS.items() for form in (verb, *forms.split())}
# Regular -ar verbs are guessed after "se"; these are not about doing math.
_NOT_MATH_VERBS = frozenset({"llama", "llaman", "pronuncia", "pronuncian"})
_MATH_TOPIC_ES = re.compile(
    r"\b(?:fraccion|ecuacion|decimal|porcentaje|por\s+ciento|razon|proporcion|exponente|potencia|raiz|raices|"
    r"radical|algebra|variable|incognita|expresion|desigualdad|inecuacion|polinomio|binomio|trinomio|monomio|"
    r"productos?\s+notables?|racional|irracional|fraccionari|logaritm|exponencial|trigonometr|factor|entero|negativo|"
    r"recta\s+numerica|grafica|pendiente|intercepto|ordenada|funcion|angulo|area|perimetro|volumen|triangulo|"
    r"circulo|circunferencia|rectangulo|cuadrado|media|mediana|moda|promedio|probabilidad|primo|multiplo|divisor|"
    r"denominador|numerador|division|multiplicacion|suma|resta|tablas?\s+de\s+multiplicar|jerarquia\s+de\s+operaciones|"
    r"orden\s+de\s+(?:las\s+)?operaciones|distributiva|terminos\s+semejantes|cuadratica|pitagoras|geometria|"
    r"problemas?\s+(?:de\s+)?(?:palabras|razonamiento|texto)|numeros?\s+mixto|valor\s+posicional|redonde|"
    r"notacion\s+cientifica|valor\s+absoluto|coordenada|plano\s+cartesiano|lineal|sistemas?\s+de|operaciones|"
    r"numero|minimo\s+comun|maximo\s+comun|mcm|mcd|regla\s+de\s+tres)\w*"
)
_DEIXIS_ES = re.compile(
    r"\b(?:esto|este|esta|estos|estas|eso|ese|esa|esos|esas|aqui|aca|ahi|alli|alla|pizarron|pizarra|arriba|abajo|"
    r"mio|mia|lo|le)\b|\b\w+(?:ar|er|ir)(?:lo|la|los|las|le|les)\b"
)
_LEADING_FILLER_ES = re.compile(
    r"^(?:mas\s+(?:sobre|de|acerca\s+de)|un\s+poco\s+(?:mas\s+)?(?:de|sobre|acerca\s+de)|un\s+poquito\s+(?:de|sobre)|"
    r"sobre|acerca\s+de|lo\s+basico\s+de|las\s+bases\s+de|lo\s+de|de|a)(?=\s|$)"
)
_TRAILING_FILLER_ES = re.compile(
    r"(?:^|\s)(?:por\s+favor|por\s+fa|porfa|porfis|plis|please|ahora|ahorita|hoy|gracias|otra\s+vez|de\s+nuevo|"
    r"para\s+mi|conmigo|profe|profesora?|maestr[oa]|ya|bien|rapido|rapidito)$"
)
# No filler is longer than this many words, so _trim only ever needs to look at that many.
_FILLER_WORDS = 6
# Nothing to teach: "quiero aprender más" / "quiero aprender lo de ayer" / "quiero aprender todo".
_VAGUE_TOPIC_ES = _word_set(
    """
    mas todo toda todos todas algo mucho muchas muchos cosa cosas otra otro nada ayer hoy manana nuevo nueva
    de lo que bien mejor el la los las del en clase clases tema temas vimos visto hicimos
    """
)


def _words(plain: str, accented: str, found: re.Match[str], group: str) -> tuple[list[str], list[str]]:
    start, end = found.span(group)
    return plain[start:end].split(), accented[start:end].split()


def _trim(
    plain: list[str], accented: list[str], pattern: re.Pattern[str], *, leading: bool
) -> tuple[list[str], list[str]]:
    start, end = 0, len(plain)
    while start < end:
        edge = plain[start : start + _FILLER_WORDS] if leading else plain[max(start, end - _FILLER_WORDS) : end]
        found = pattern.search(" ".join(edge))
        count = len(found.group().split()) if found else 0
        if not count:
            break
        if leading:
            start += count
        else:
            end -= count
    return plain[start:end], accented[start:end]


def _finish(plain: list[str], accented: list[str], verb: str | None = None) -> str | None:
    plain, accented = _trim(plain, accented, _TRAILING_FILLER_ES, leading=False)
    plain, accented = plain[:MAX_TOPIC_WORDS], accented[:MAX_TOPIC_WORDS]
    if not plain or _VAGUE_TOPIC_ES.issuperset(plain) or _DEIXIS_ES.search(" ".join(plain)):
        return None
    if verb:
        accented = [verb, *accented[1:]]
    return " ".join(accented)


def _infinitive(word: str, *, guess: bool) -> str | None:
    if word in _INFINITIVES:
        return _INFINITIVES[word]
    if not guess or len(word) < 4 or word in _NOT_MATH_VERBS:
        return None
    if word.endswith("an"):
        return word[:-2] + "ar"
    if word.endswith("a"):
        return word[:-1] + "ar"
    return None


def _howto(plain: str, accented: str, *, guarded: bool) -> str | None:
    """Topic of "¿cómo se suman fracciones?" / "¿cómo hago para ...?", as an infinitive phrase ("sumar fracciones")."""
    for kind, pattern in _HOWTO_ES:
        found = pattern.search(plain)
        if not found:
            continue
        verb = None
        if kind != "infinitive":
            verb = _infinitive(found.group("verb"), guess=kind == "impersonal")
            if verb is None:
                continue
        words_plain, words_accented = _words(plain, accented, found, "phrase")
        # The verb itself is no topic: "¿cómo se suma 2/3 + 1/4?" is about the board.
        if guarded and not _MATH_TOPIC_ES.search(" ".join(words_plain[1:])):
            return None
        return _finish(words_plain, words_accented, verb)
    return None


def _topic_es(plain: list[str], accented: list[str], *, guarded: bool) -> str | None:
    plain, accented = _trim(plain, accented, _LEADING_FILLER_ES, leading=True)
    if plain[:1] == ["como"]:
        return _howto(" ".join(plain), " ".join(accented), guarded=guarded)
    if guarded and not _MATH_TOPIC_ES.search(" ".join(plain)):
        return None
    return _finish(plain, accented)


def detect_learn_request_es(text: str) -> str | None:
    """The topic, in Spanish, of "quiero aprender ..." / "enséñame ..." / "¿cómo se ...?", else None."""
    plain, accented = _normalize(text)
    for pattern in _LEARN_TRUSTED_ES:
        for found in pattern.finditer(plain):
            if not _negated(plain, found.start()):
                return _topic_es(*_words(plain, accented, found, "rest"), guarded=False)
    found = _LEARN_EXPLAIN_ES.search(plain)
    if found:
        return _topic_es(*_words(plain, accented, found, "rest"), guarded=True)
    return _howto(plain, accented, guarded=True)


# ---- Spanish intents: "¿tienes un ejercicio o invento uno?" ----------------------
# A bare "uno"/"una" only counts at the end or before a few fillers: "tengo una duda" is not an exercise, and
# neither is "tengo una bien buena pregunta", so "bien" / "fácil" must lead to the end or another filler.
_ONE_OWN = (
    r"(?:uno|una)(?=(?:\s+(?:bien|muy|super|facil|dificil|facilito|facilita|dificilito|dificilita|sencillo|"
    r"sencilla|complicado|complicada))*(?:\s*(?:,|$)|\s+(?:aqui|aca|ya|listo|lista|en|de|que|mio|mia|propio|"
    r"propia|pero|y)\b))"
)
_ONE_ASKED = (
    r"(?:uno|una|otro|otra)(?=\s*(?:,|$)|\s+(?:tu|ya|por|porfa|porfis|plis|de|para|facil|dificil|nuevo|nueva|que|"
    r"mas|cualquiera|rapido)\b)"
)
_MINE = r"(?:el|la|los|las)\s+mi[oa]s?\b"
_OFFER_DECLINE_ES = _compile(
    r"\bno\s+(?:tengo|traigo|tenemos)\b",
    r"\bno\s+se\s+me\s+ocurre\b",
    r"\bno\s+(?:lo|la|los|las)\s+(?:tengo|traigo)(?=\s*(?:,|$)|\s+(?:aqui|aca|conmigo|ahorita|a\s+la\s+mano)\b)",
)
_OFFER_OWN_ES = _compile(
    rf"\b(?:tengo|traigo)\s+(?:{_ONE_OWN}|mis?\s+(?:propi[oa]s?|ejercicios?|problemas?|tarea|ecuacion(?:es)?|ejemplos?)\b|"
    rf"un\s+(?:ejercicio|problema|ejemplo)\b|una\s+(?:ecuacion|fraccion|operacion)\b|tarea\b|{_MINE})",
    r"\b(?:lo\s+|la\s+)?(?:voy|vamos)\s+a\s+(?:escribir|poner|dibujar|anotar|usar)(?:lo|la)?\b",
    r"\b(?:dejame|deja\s+me|dejeme)\s+(?:escribir|poner|dibujar|anotar|mostrar|ensenar)\w*",
    rf"\b{_MINE}",
    r"\bmis?\s+propi[oa]s?\b",
    r"\bde\s+(?:la|mi)\s+tarea\b",
    r"\bde\s+mi\s+(?:libro|cuaderno|guia|hoja)\b",
    # Not "yo te digo una cosa".
    rf"\byo\s+te\s+(?:doy|digo|dicto|paso|pongo|escribo|invento)\s+{_ONE_OWN}",
    r"\bya\s+(?:lo|la)\s+(?:escribi|puse|anote|dibuje)\b",
    r"\bahi\s+te\s+va\b",
)
# "yo lo escribo" / "lo hago yo" offer one; "no lo hago yo" and "yo lo hago mal" do not.
_OFFER_OWN_DO_ES = _compile(
    r"\b(?:yo\s+)?(?:lo\s+|la\s+|uno\s+)?(?:escribo|pongo|dibujo|anoto|invento)\b(?!\s+mal\b)",
    r"\b(?:yo\s+(?:lo\s+|la\s+|uno\s+)?hago|lo\s+hago\s+yo|hago\s+uno\s+yo)\b(?!\s+mal\b)",
)
_REFUSED_ES = re.compile(r"\b(?:no|nunca|tampoco)\s+(?:(?:yo|lo|la|uno)\s+)*$")
_OFFER_OWN_ALONE_ES = re.compile(
    r"^(?:(?:(?:si|sip|simon|claro|aja)\s+)*(?:yo\s+)?(?:si\s+)?(?:tengo|traigo)|(?:no\s+)?(?:mejor\s+yo|yo\s+mejor))$"
)
_OFFER_GENERATE_ES = _compile(
    r"\binventa(?:me|te|nos|lo|melo)?\b",
    r"\b(?:dame|danos|hazme|haznos|ponme|ponnos|escribeme|creame|haz|pon|crea|escribe|mandame|echame|avientame|"
    rf"pasame)\s+(?:tu\s+)?(?:{_ONE_ASKED}|cualquiera\b|(?:un|otro)\s+(?:ejercicio|problema|ejemplo|reto)\b|"
    r"(?:una|otra)\s+(?:ecuacion|fraccion|operacion)\b)",
    r"\b(?:me|nos)\s+(?:das|haces|pones|inventas|escribes|(?:puedes|podrias)\s+(?:dar|hacer|poner|inventar))\s+"
    rf"(?:{_ONE_ASKED}|un\s+(?:ejercicio|problema|ejemplo)\b|una\s+(?:ecuacion|fraccion)\b)",
    r"\btu\s+(?:escoge|escoges|elige|eliges|decide|decides|hazlo|haz\s+uno|inventa\w*|ponlo|pon\s+uno|lo\s+haces|"
    r"lo\s+inventas|lo\s+eliges|lo\s+escoges|dime)\b",
    r"\b(?:escoge|elige|decide|escogelo|eligelo|inventalo|hazlo|ponlo)\s+tu\b",
    r"\b(?:como|lo\s+que|el\s+que|la\s+que)\s+(?:tu\s+)?(?:quieras|digas|gustes)\b",
    r"\b(?:lo|el|la)\s+que\s+sea\b",
    r"\b(?:uno|una)\s+tuy[oa]\b",
    r"\bme\s+da\s+(?:igual|lo\s+mismo)\b",
    r"\bsorprendeme\b",
)
_OFFER_GENERATE_ALONE_ES = re.compile(
    r"^(?:no|nop|nope|nel|nel\s+pastel|no\s+gracias|pues\s+no|la\s+verdad\s+no|no\s+no|(?:no\s+)?(?:ninguno|ninguna|nada)|"
    r"cualquiera|(?:(?:no|nel|pues)\s+)?(?:mejor\s+tu|tu))(?:\s+(?:porfa|porfis|plis|por\s+favor|gracias))*$"
)
# "uno fácil" / "sí, uno de fracciones": the kind of exercise to invent, as the whole clause.
_OFFER_ONE_ALONE_ES = re.compile(
    rf"(?:(?:si|sip|simon|ok|okay|va|sale|dale|bueno|pues|mejor)\s+)*(?:uno|una|otro|otra)(?:\s+(?:{_HARDER}|facil|"
    r"dificil|facilito|facilita|nuevo|nueva|rapido|sencillo|sencilla|cortito|cortita|"
    # "uno de tarea" / "uno de mi libro" may be the student's own.
    r"de\s+(?!(?:tarea|mi|mis|la|las|los|el)\b)\w+))+"
    r"(?:\s+(?:porfa|porfis|plis|por\s+favor|gracias))*"
)


def _offered(plain: str) -> bool:
    return any(
        not _REFUSED_ES.search(plain, max(0, m.start() - _LOOKBACK), m.start())
        for p in _OFFER_OWN_DO_ES
        for m in p.finditer(plain)
    )


def detect_offer_reply_es(text: str) -> str | None:
    """Answer to "¿Tienes un ejercicio en mente o invento uno?": "generate", "own" or None."""
    plain = _normalize(text)[0]
    flat = _flat(plain)
    if _any(_OFFER_DECLINE_ES, plain):
        return "generate"
    if _any(_OFFER_OWN_ES, plain) or _offered(plain) or _OFFER_OWN_ALONE_ES.search(flat):
        return "own"
    if _any(_OFFER_GENERATE_ES, plain) or _OFFER_GENERATE_ALONE_ES.search(flat) or _alone(_OFFER_ONE_ALONE_ES, plain):
        return "generate"
    return None


# ---- both languages ------------------------------------------------------------
_TUTOR = "biobench_sar.tutor"


class _EnglishDetectors:
    """tutor.py's English detectors, imported late: tutor may import this module, and may not exist yet.

    A missing tutor is remembered; a tutor still half-imported (it imports this module) is looked at again next
    call; a tutor that fails to import for any other reason is logged, not silently treated as missing.
    """

    def __init__(self) -> None:
        self._detectors: tuple[Detector, Detector, Detector] | None = None
        self._unavailable = False

    def __call__(self) -> tuple[Detector, Detector, Detector] | None:
        if self._detectors is None and not self._unavailable:
            self._load()
        return self._detectors

    def cache_clear(self) -> None:
        self._detectors, self._unavailable = None, False

    def _load(self) -> None:
        try:
            tutor = importlib.import_module(_TUTOR)
        except ImportError as error:
            if not (isinstance(error, ModuleNotFoundError) and error.name == _TUTOR):
                log.warning("English intent detection is off: %s failed to import", _TUTOR, exc_info=True)
            self._unavailable = True
            return
        mode = getattr(tutor, "detect_mode", None)
        learn = getattr(tutor, "detect_learn_request", None)
        offer = getattr(tutor, "detect_offer_reply", None)
        if callable(mode) and callable(learn) and callable(offer):
            self._detectors = (mode, learn, offer)


_english_detectors = _EnglishDetectors()


_SPANISH_DETECTORS: tuple[Detector, Detector, Detector] = (
    detect_mode_es,
    detect_learn_request_es,
    detect_offer_reply_es,
)
_INTENTS = ("mode", "learn_topic", "offer_reply")


def detect_all(text: str, default: str = "en") -> dict[str, str | None]:
    """{"language", "mode", "learn_topic", "offer_reply"}; each intent is read in the detected language first.

    Only the first MAX_INPUT_CHARS characters are read, by the English detectors too: a request after that is
    not seen.
    """
    text = (text or "")[:MAX_INPUT_CHARS]
    language = detect_language(text, default)
    english = _english_detectors()
    order = (_SPANISH_DETECTORS, english) if language == "es" else (english, _SPANISH_DETECTORS)
    found: dict[str, str | None] = {"language": language}
    for index, key in enumerate(_INTENTS):
        found[key] = None
        for detectors in order:
            value = detectors[index](text) if detectors else None
            if value is not None:
                found[key] = value
                break
    return found
