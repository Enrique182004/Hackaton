"""Bilingual (en/es) helpers: which language the student is using, and the Spanish tutor intents.

The Spanish detectors return exactly what tutor.py's English ones return, so a caller can try both.
"""

from __future__ import annotations

import functools
import importlib
import re
import unicodedata
from collections.abc import Callable
from typing import Literal

Language = Literal["en", "es"]
LanguageSetting = Literal["auto", "en", "es"]
LANGUAGES: tuple[Language, ...] = ("en", "es")
MAX_TOPIC_WORDS = 8

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


def normalize_language(code: str | None) -> LanguageSetting:
    """"en" / "es" from a setting or locale ("es-MX", "English", "Español"); anything else is "auto"."""
    key = _plain((code or "").strip().lower()).replace("_", "-").split("-")[0]
    if key in _EN_NAMES:
        return "en"
    if key in _ES_NAMES:
        return "es"
    return "auto"


# ---- language detection ------------------------------------------------------
# Accent-free, and never in both sets: words shared by the two languages ("no", "me", "a", "quiz", "total")
# or that are math in English ("sin", "tan", "dice") vote for neither.
_ES_WORDS = frozenset(
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
    atorada
    paso pasos problema problemas ecuacion ecuaciones fraccion fracciones ejercicio ejercicios respuesta
    respuestas pregunta preguntas numero numeros resultado raiz cuadrado cubo tarea examen prueba pizarron pizarra
    hoja ejemplo tema clase lado parte igual mitad medio tercio tercios cuarto cuartos vez veces punto signo valor
    letra pista practica porcentaje ciento grafica linea recta angulo triangulo perimetro duda maestra maestro profe
    cero uno dos tres cuatro cinco seis siete ocho nueve diez doce trece catorce quince veinte treinta cuarenta
    cincuenta cien mil
    hola gracias oye mande porfa vale orale andale chido listo lista claro adios
    """.split()
)
_EN_WORDS = frozenset(
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
    write draw board back way side value whole
    """.split()
)
_SPANISH_PUNCT = re.compile(r"[¿¡]")
_SPANISH_LETTERS = re.compile(r"[áéíóúüñ]")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_MATH_CHARS = frozenset("0123456789=+-−*/^()[]{}<>≤≥≠×÷·√%.")


def _is_variable(text: str, start: int, end: int) -> bool:
    """A lone letter beside numbers or operators ("y = 2x", "3 y 4"), or on its own, is math rather than a word."""
    before = text[:start].rstrip()[-1:]
    after = text[end:].lstrip()[:1]
    return (not before and not after) or before in _MATH_CHARS or after in _MATH_CHARS


def _vote(word: str, previous: str | None, previous_vote: str | None) -> str | None:
    if _plain(word) in _ES_WORDS:
        return "es"
    if word in _EN_WORDS:
        return "en"
    if word == "me":  # "help me" / "quiz me"; in "¿me ayudas?" the next word decides
        return "en" if previous == "quiz" else previous_vote
    return None


def detect_language(text: str, default: str = "en") -> Language:
    """"en" or "es" for what the student said or typed; `default` when nothing leans either way (math only, "ok")."""
    fallback: Language = "es" if normalize_language(default) == "es" else "en"
    lowered = unicodedata.normalize("NFC", (text or "").lower()).replace("’", "'").replace("‘", "'")
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
    lowered = unicodedata.normalize("NFC", (text or "").lower()).replace("’", "'").replace("‘", "'")
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


# What comes right before a command that makes it not one: negated ("no quiero un quiz"), asked about
# ("¿cuándo termina el quiz?") or a purpose clause ("falta poco para terminar el quiz").
_NOT_A_COMMAND = re.compile(
    r"(?:\b(?:no|nunca|ni|tampoco)\s+"
    r"(?:(?:quiero|queria|quisiera|queremos|puedo|podemos|puedes|necesito|debo|debemos|voy\s+a|vamos\s+a|"
    r"tengo\s+que|hay\s+que|me\s+gusta|me\s+gustaria)\s+)?"
    r"|\b(?:cuando|como|donde|cuanto|cuanta|cuantos|cuantas|cual|cuales|quien|porque|por\s+que|para\s+que|"
    r"que\s+hora|para|antes\s+de|despues\s+de|hasta|sin|al|se)\s+)$"
)


def _commanded(patterns: tuple[re.Pattern[str], ...], plain: str) -> bool:
    return any(not _NOT_A_COMMAND.search(plain, 0, m.start()) for p in patterns for m in p.finditer(plain))


# ---- Spanish intents: mode -----------------------------------------------------
_QUIZ_WORD = r"(?:quiz|quizz|prueba|examen|examencito|evaluacion|cuestionario)"
_THE_QUIZ = rf"(?:ya\s+)?(?:de\s+|con\s+)?(?:el\s+|este\s+|ese\s+|mi\s+|del\s+)?{_QUIZ_WORD}\b"

# "para" / "termina" / "acaba" are also "for" / "it ends": only commands when they open the sentence.
_STOP_OPENING_ES = re.compile(
    r"^(?:(?:ya|oye|mejor|bueno|ok|okay|ahora|pues|porfa|por\s+favor|profe|maestra|maestro|no)\s+)*"
    rf"(?:para|parale|parele|termina|terminale|acaba|acabale|acaba\s+con|termina\s+con)\s+{_THE_QUIZ}"
)
_STOP_ES = _compile(
    rf"\b(?:terminemos|paremos|detengamos|cancelemos|salgamos|acabemos|dejemos|finalicemos|cerremos|quitemos)\s+{_THE_QUIZ}",
    rf"\b(?:terminar|parar|detener|cancelar|salir|salirme|acabar|dejar|finalizar|cerrar|quitar)\s+{_THE_QUIZ}",
    rf"\b(?:cancela|deten|detente|quita|cierra|finaliza|sal|salte)\s+{_THE_QUIZ}",
    rf"\bya\s+no\s+(?:quiero|queremos|quiere)\s+(?:hacer\s+|seguir\s+|con\s+|el\s+|un\s+|este\s+|mas\s+)*{_QUIZ_WORD}\b",
    r"\bno\s+mas\s+(?:quiz|quizz|quizzes|examen|examenes|prueba|pruebas)\b",
    rf"\b(?:basta|suficiente)\s+(?:de|con)\s+(?:el\s+|este\s+)?{_QUIZ_WORD}\b",
)
_QUIZ_ES = _compile(
    r"\b(?:examiname|evaluame|examinanos|evaluanos)\b",
    r"\bponme\s+a\s+prueba\b",
    r"\b(?:hazme|haznos|dame|danos|ponme|ponnos|aplicame|preparame|haz|pon|prepara)\s+"
    rf"(?:un|una|el|otro|otra)\s+(?:\w+\s+)?{_QUIZ_WORD}\b",
    r"\b(?:me|nos)\s+(?:haces|hagas|harias|puedes\s+hacer|podrias\s+hacer|pones|podrias\s+poner|das)\s+"
    rf"(?:un|una|otro|otra)\s+(?:\w+\s+)?{_QUIZ_WORD}\b",
    r"\b(?:quiero|queremos|quisiera|queria|me\s+gustaria|necesito)\s+(?:hacer\s+|tomar\s+|presentar\s+|tener\s+)?"
    rf"(?:un|una|el|otro|otra)\s+(?:\w+\s+)?{_QUIZ_WORD}\b",
    r"\b(?:empecemos|empieza|empezar|comienza|comenzar|comencemos|inicia|iniciar|iniciemos|hagamos|hacemos|"
    rf"arranca|arranquemos|vamos\s+a\s+hacer)\s+(?:con\s+)?(?:un\s+|una\s+|el\s+|otro\s+|otra\s+)?{_QUIZ_WORD}\b",
    rf"\bmodo\s+(?:de\s+)?{_QUIZ_WORD}\b",
)
_LIKE_THIS = r"(?:parecid[oa]s?|similar(?:es)?|asi|como\s+(?:este|esta|estos|estas|ese|esa|esto|eso|el\s+anterior|la\s+anterior))"
_EXERCISE = r"(?:problema|ejercicio|ejemplo|pregunta|ecuacion|fraccion|operacion)"
_PRACTICE_ES = _compile(
    # "uno igual" is left out: "dos más uno igual a tres" is dictation, not a request.
    rf"\b(?:otro|otra)\s+(?:mas\s+)?(?:{_EXERCISE}\s+)?(?:mas\s+)?(?:{_LIKE_THIS}|igual(?:ito|ita)?)\b",
    rf"\b(?:uno|una)\s+(?:mas\s+)?{_LIKE_THIS}\b",
    rf"\b(?:un|una)\s+{_EXERCISE}\s+{_LIKE_THIS}\b",
    rf"\b{_EXERCISE}s\s+(?:mas\s+)?{_LIKE_THIS}\b",
    r"\bmas\s+practica\b",
    r"\b(?:practicar|practiquemos)\s+(?:mas|otro|otra|un\s+poco)\b",
    r"\bquiero\s+practicar\b",
    r"\bmodo\s+(?:de\s+)?practica\b",
    r"\b(?:problemas|ejercicios)\s+de\s+practica\b",
)
_HELP_ES = _compile(
    r"\bayud(?:a|ame|arme|as|en|enme|ita|eme)\b",
    r"\bauxilio\b",
    r"\b(?:siguiente|proximo)\s+paso\b",
    r"\bque\s+(?:sigue|hago|debo\s+hacer|tengo\s+que\s+hacer|hay\s+que\s+hacer|mas\s+hago|le\s+hago)\b",
    r"\by\s+ahora\s+(?:que|como)\b",
    r"\bno\s+(?:le\s+|lo\s+|la\s+)?(?:entiendo|entendi|comprendo|comprendi|capto|cacho)\b",
    r"\bno\s+se\s+(?:como|que\s+hacer|que\s+sigue|que\s+hago|por\s+donde)\b",
    r"\b(?:estoy|ando|me\s+quede|me\s+siento|sigo|quede)\s+(?:bien\s+|muy\s+|super\s+|todo\s+|bastante\s+)?"
    r"(?:atorad[oa]|atascad[oa]|trabad[oa]|perdid[oa]|confundid[oa]|stuck)\b",
    r"\bme\s+(?:atore|atasque|trabe|perdi|confundi|revolvi)\b",
    r"\bpista\b",
)
_HELP_ALONE_ES = re.compile(r"^y\s+(?:ahora|luego|despues|entonces)(?:\s+que)?$")
_NO_HELP_ES = re.compile(r"\bno\s+(?:necesito|ocupo|quiero)\s+(?:tu\s+|mas\s+|nada\s+de\s+)?ayuda\b")


def _is_quiz_stop(plain: str) -> bool:
    return bool(_STOP_OPENING_ES.search(_flat(plain))) or _commanded(_STOP_ES, plain)


def is_quiz_stop_es(text: str) -> bool:
    """An explicit "termina el quiz": "ayuda" / "estoy atorado" during a quiz asks for a hint, not to quit."""
    return _is_quiz_stop(_normalize(text)[0])


def detect_mode_es(text: str) -> str | None:
    """"help" | "practice" | "quiz" a student's Spanish asks for, or None to stay in the current mode."""
    plain = _normalize(text)[0]
    if _is_quiz_stop(plain):
        return "help"
    if _commanded(_QUIZ_ES, plain):
        return "quiz"
    if _commanded(_PRACTICE_ES, plain):
        return "practice"
    if _NO_HELP_ES.search(plain):
        return None
    if _any(_HELP_ES, plain) or _HELP_ALONE_ES.search(_flat(plain)):
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
    ("infinitive", re.compile(r"\bcomo\s+(?:se\s+)?(?:le\s+)?(?:hago|hace|hacemos|haces|hacen|hacer)\s+para\s+(?P<phrase>[^,]+)")),
    (
        "infinitive",
        re.compile(
            r"\bcomo\s+(?:se\s+)?(?:le\s+)?(?:puedo|podemos|puede|puedes|podria|podriamos|debo|debemos|debe|"
            r"tengo\s+que|tenemos\s+que|tiene\s+que|hay\s+que)\s+(?P<phrase>[^,]+)"
        ),
    ),
    ("impersonal", re.compile(r"\bcomo\s+se\s+(?P<phrase>(?P<verb>[a-z]+)\s+[^,]+)")),
    ("personal", re.compile(r"\bcomo\s+(?P<phrase>(?P<verb>[a-z]+)\s+[^,]+)")),
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
    r"radical|algebra|variable|incognita|expresion|desigualdad|inecuacion|polinomio|factor|entero|negativo|"
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


def _words(plain: str, accented: str, found: re.Match[str], group: str) -> tuple[list[str], list[str]]:
    start, end = found.span(group)
    return plain[start:end].split(), accented[start:end].split()


def _trim(
    plain: list[str], accented: list[str], pattern: re.Pattern[str], *, leading: bool
) -> tuple[list[str], list[str]]:
    while plain and (found := pattern.search(" ".join(plain))):
        count = len(found.group().split())
        if not count:
            break
        plain, accented = (plain[count:], accented[count:]) if leading else (plain[:-count], accented[:-count])
    return plain, accented


def _finish(plain: list[str], accented: list[str], verb: str | None = None) -> str | None:
    plain, accented = _trim(plain, accented, _TRAILING_FILLER_ES, leading=False)
    plain, accented = plain[:MAX_TOPIC_WORDS], accented[:MAX_TOPIC_WORDS]
    if not plain or _DEIXIS_ES.search(" ".join(plain)):
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
            if not _NOT_A_COMMAND.search(plain, 0, found.start()):
                return _topic_es(*_words(plain, accented, found, "rest"), guarded=False)
    found = _LEARN_EXPLAIN_ES.search(plain)
    if found:
        return _topic_es(*_words(plain, accented, found, "rest"), guarded=True)
    return _howto(plain, accented, guarded=True)


# ---- Spanish intents: "¿tienes un ejercicio o invento uno?" ----------------------
# A bare "uno"/"una" only counts at the end or before a few fillers: "tengo una duda" is not an exercise.
_ONE_OWN = r"(?:uno|una)(?=\s*(?:,|$)|\s+(?:aqui|aca|ya|listo|lista|en|de|que|mio|mia|propio|propia)\b)"
_ONE_ASKED = (
    r"(?:uno|una|otro|otra)(?=\s*(?:,|$)|\s+(?:tu|ya|por|porfa|porfis|plis|de|para|facil|dificil|nuevo|nueva|que|"
    r"mas|cualquiera|rapido)\b)"
)
_OFFER_DECLINE_ES = _compile(r"\bno\s+(?:tengo|traigo|tenemos)\b", r"\bno\s+se\s+me\s+ocurre\b")
_OFFER_OWN_ES = _compile(
    rf"\b(?:tengo|traigo)\s+(?:{_ONE_OWN}|mis?\s+(?:propi[oa]s?|ejercicios?|problemas?|tarea|ecuacion(?:es)?|ejemplos?)\b|"
    r"un\s+(?:ejercicio|problema|ejemplo)\b|una\s+(?:ecuacion|fraccion|operacion)\b|tarea\b|el\s+mio\b|la\s+mia\b)",
    r"\b(?:lo\s+|la\s+)?(?:voy|vamos)\s+a\s+(?:escribir|poner|dibujar|anotar|usar)(?:lo|la)?\b",
    r"\b(?:dejame|deja\s+me|dejeme)\s+(?:escribir|poner|dibujar|anotar|mostrar|ensenar)\w*",
    r"\b(?:yo\s+)?(?:lo\s+|la\s+|uno\s+)?(?:escribo|pongo|dibujo|anoto|invento)\b",
    r"\b(?:el|la)\s+mi[oa]\b",
    r"\bmi\s+propi[oa]\b",
    r"\bde\s+mi\s+tarea\b",
)
_OFFER_OWN_ALONE_ES = re.compile(r"^(?:(?:si|sip|simon|claro|aja)\s+)*(?:yo\s+)?(?:si\s+)?tengo$")
_OFFER_GENERATE_ES = _compile(
    r"\binventa(?:me|te|nos|lo|melo)?\b",
    r"\b(?:dame|danos|hazme|haznos|ponme|ponnos|escribeme|creame|haz|pon|crea|escribe)\s+(?:tu\s+)?"
    rf"(?:{_ONE_ASKED}|un\s+(?:ejercicio|problema|ejemplo|reto)\b|una\s+(?:ecuacion|fraccion|operacion)\b)",
    r"\b(?:me|nos)\s+(?:das|haces|pones|inventas|escribes|(?:puedes|podrias)\s+(?:dar|hacer|poner|inventar))\s+"
    rf"(?:{_ONE_ASKED}|un\s+(?:ejercicio|problema|ejemplo)\b|una\s+(?:ecuacion|fraccion)\b)",
    r"\btu\s+(?:escoge|escoges|elige|eliges|decide|decides|hazlo|haz\s+uno|inventa\w*|ponlo|pon\s+uno|lo\s+haces|"
    r"lo\s+inventas|lo\s+eliges|lo\s+escoges|dime)\b",
    r"\b(?:escoge|elige|decide|escogelo|eligelo|inventalo|hazlo|ponlo)\s+tu\b",
    r"\b(?:como|lo\s+que)\s+(?:tu\s+)?quieras\b",
    r"\bsorprendeme\b",
)
_OFFER_NO_ALONE_ES = re.compile(
    r"^(?:no|nop|nope|nel|nel\s+pastel|no\s+gracias|pues\s+no|la\s+verdad\s+no|no\s+no|ninguno|nada|no\s+nada)$"
)


def detect_offer_reply_es(text: str) -> str | None:
    """Answer to "¿Tienes un ejercicio en mente o invento uno?": "generate", "own" or None."""
    plain = _normalize(text)[0]
    flat = _flat(plain)
    if _any(_OFFER_DECLINE_ES, plain):
        return "generate"
    if _any(_OFFER_OWN_ES, plain) or _OFFER_OWN_ALONE_ES.search(flat):
        return "own"
    if _any(_OFFER_GENERATE_ES, plain) or _OFFER_NO_ALONE_ES.search(flat):
        return "generate"
    return None


# ---- both languages ------------------------------------------------------------
@functools.cache
def _english_detectors() -> tuple[Detector, Detector, Detector] | None:
    """tutor.py's English detectors, imported late: tutor may import this module, and may not exist yet."""
    try:
        tutor = importlib.import_module("biobench_sar.tutor")
    except ImportError:
        return None
    mode = getattr(tutor, "detect_mode", None)
    learn = getattr(tutor, "detect_learn_request", None)
    offer = getattr(tutor, "detect_offer_reply", None)
    if not (callable(mode) and callable(learn) and callable(offer)):
        return None
    return mode, learn, offer


_SPANISH_DETECTORS: tuple[Detector, Detector, Detector] = (
    detect_mode_es,
    detect_learn_request_es,
    detect_offer_reply_es,
)
_INTENTS = ("mode", "learn_topic", "offer_reply")


def detect_all(text: str, default: str = "en") -> dict[str, str | None]:
    """{"language", "mode", "learn_topic", "offer_reply"}; each intent is read in the detected language first."""
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
