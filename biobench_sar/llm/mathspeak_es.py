"""Spanish (es-MX) phonetic math for TTS: "3x+5=12" -> "tres x más cinco es igual a doce"."""

from __future__ import annotations

import re

__all__ = ["fraction_to_words_es", "has_math_symbols", "number_to_words_es", "to_speech_es"]

# Private-use sentinels carry structure between passes; every one becomes words or a soft space at the end.
LQ, RQ = "\ue000", "\ue001"  # (x+1)  -> "la cantidad x más uno"
LS, RS = "\ue002", "\ue003"  # (3)    -> single term, parens dropped
LF, RF = "\ue004", "\ue005"  # f(x)   -> "f de x"
LE, RE = "\ue006", "\ue007"  # x^(n+1) -> compound exponent
OS, OE = "\ue008", "\ue009"  # an operand already spelled out (still an operand for minus / implicit-× decisions)
TIMES = "\ue00a"
SOFT = "\ue00b"  # a space we introduced: collapsible, dropped next to punctuation
PL, PR = "\ue00c", "\ue00d"  # prose parens parked while grouping
YV = "\ue00e"  # the variable y, read "ye" (a bare "y" is the conjunction "i" to a Spanish voice)
_CLOSERS = RQ + RS + RF + RE + OE
_OPENERS = LQ + LS + LF + LE
_OPS = "=+-*/×·÷<>≤≥≠≈±^"

# fmt: off
_UNITS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
    "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno",
    "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
    "veintinueve",
]
# fmt: on
_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_HUNDREDS = {
    1: "ciento",
    2: "doscientos",
    3: "trescientos",
    4: "cuatrocientos",
    5: "quinientos",
    6: "seiscientos",
    7: "setecientos",
    8: "ochocientos",
    9: "novecientos",
}
_DENOMS = {
    2: "medio",
    3: "tercio",
    4: "cuarto",
    5: "quinto",
    6: "sexto",
    7: "séptimo",
    8: "octavo",
    9: "noveno",
    10: "décimo",
    21: "veintiunavo",
    100: "centésimo",
    1000: "milésimo",
}
_POWER_ORDINALS = {
    0: "cero",
    1: "primera",
    4: "cuarta",
    5: "quinta",
    6: "sexta",
    7: "séptima",
    8: "octava",
    9: "novena",
    10: "décima",
}
# n -> (masculine, feminine, apocope before a noun, accepted written suffixes)
_ORDINALS = {
    1: ("primero", "primera", "primer", ("er", "ero", "era", "ro", "ra")),
    2: ("segundo", "segunda", "segundo", ("do", "da")),
    3: ("tercero", "tercera", "tercer", ("er", "ero", "era", "ro", "ra")),
    4: ("cuarto", "cuarta", "cuarto", ("to", "ta")),
    5: ("quinto", "quinta", "quinto", ("to", "ta")),
    6: ("sexto", "sexta", "sexto", ("to", "ta")),
    7: ("séptimo", "séptima", "séptimo", ("mo", "ma")),
    8: ("octavo", "octava", "octavo", ("vo", "va")),
    9: ("noveno", "novena", "noveno", ("no", "na")),
    10: ("décimo", "décima", "décimo", ("mo", "ma")),
}
_UNICODE_FRACTIONS = {
    "½": "1/2",
    "⅓": "1/3",
    "⅔": "2/3",
    "¼": "1/4",
    "¾": "3/4",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻ⁿ", "0123456789-n")
_UNACCENT = str.maketrans("áéíóú", "aeiou")
_OPERATOR_WORDS = {
    "+": "más",
    "=": "es igual a",
    "<": "es menor que",
    ">": "es mayor que",
    "≤": "es menor o igual que",
    "≥": "es mayor o igual que",
    "≠": "es diferente de",
    "≈": "es aproximadamente igual a",
    "±": "más o menos",
    "×": "por",
    "÷": "entre",
    "->": "entonces",
    "=>": "entonces",
    "→": "entonces",
    "⇒": "entonces",
}
_ROOTS = {"√": "la raíz cuadrada de", "∛": "la raíz cúbica de", "∜": "la raíz cuarta de"}

# Gender of the noun after a number decides "un paso" / "una vez" / "doscientas manzanas".
# fmt: off
_MASC_A = frozenset([
    "problema", "sistema", "tema", "día", "mapa", "programa", "teorema", "diagrama", "esquema", "idioma", "planeta",
    "clima", "poema", "dilema", "enigma",
])
_FEM_OTHER = frozenset([
    "vez", "veces", "clase", "parte", "mano", "raíz", "raíces", "luz", "luces", "imagen", "imágenes", "base", "fase",
    "noche", "tarde", "ley", "red",
])
_NOT_NOUNS = frozenset([
    "de", "del", "y", "e", "o", "u", "en", "a", "al", "es", "son", "era", "más", "menos", "por", "entre", "sobre",
    "que", "con", "para", "sin", "la", "las", "el", "los", "lo", "le", "se", "no", "sí", "si", "como", "ni", "pero",
    "porque", "cuando", "hasta", "desde", "ya", "también", "resta", "suma", "multiplica", "divide", "despeja",
    "simplifica", "escribe", "piensa", "revisa", "intenta", "usa", "mueve", "pasa", "cambia", "agrupa", "factoriza",
    "sustituye", "calcula", "comprueba", "verifica", "elige", "dibuja", "cuenta", "mira", "nota", "recuerda",
])
# fmt: on

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_COORD = r"-?[ \t]*(?:\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z])"
_SYMBOLS = "^/*=+<>×·÷√∛∜≤≥≠≈±%π∞→⇒⁄∕⁰¹²³⁴⁵⁶⁷⁸⁹⁻" + "".join(_UNICODE_FRACTIONS)
_MATH_HINT = re.compile(r"[\d" + _SYMBOLS + r"]|-(?=\d)|(?<![^\W\d_])[a-zA-Z][ \t]*-[ \t]*[a-zA-Z(](?![^\W\d_])")
_MAYBE_MATH = re.compile(r"[\d" + _SYMBOLS + r"(\-]")
_NORMALIZE = (
    (re.compile("[\ue000-\uf8ff]"), ""),
    (re.compile("[−﹣－]"), "-"),
    (re.compile("[⁄∕]"), "/"),
    (re.compile("<="), "≤"),
    (re.compile(">="), "≥"),
    (re.compile("!=|=/="), "≠"),
    (re.compile("={2,3}"), "="),
    (re.compile(r"(?<=[\w)])\*\*(?=[\w(])"), "^"),
    (re.compile(r"\bsqrt\s*(?=\()", re.I), "√"),
    (re.compile(r"\bcbrt\s*(?=\()", re.I), "∛"),
    (re.compile(r"\by/o\b"), "y o"),
    (re.compile(r"\b(ra[ií]z)(?:\s+cuadrada)?(?:\s+de)?\s*√", re.I), r"\1 cuadrada de "),
)
_SUPERSCRIPT_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻ⁿ]+")
_UNICODE_FRACTION = re.compile("(\\d?)([" + "".join(_UNICODE_FRACTIONS) + "])")
_POINT = re.compile(
    rf"(?P<pre>\b(?:punto|puntos|coordenadas?)[ \t]*)?\([ \t]*(?P<a>{_COORD})[ \t]*,[ \t]*(?P<b>{_COORD})[ \t]*\)",
    re.I,
)
_GROUP = re.compile(r"\(([^()]*)\)")
_MATH_CONTENT = re.compile("[\\s\\d.a-zA-Z+\\-*/×·÷=<>≤≥≠≈±^√∛∜%π\ue000-\ue00b]*")
_TERM = re.compile(r"\s*-?\s*(?:\d+(?:\.\d+)?(?:/\d+)?[a-zA-Z]?|[a-zA-Z]|π)\s*")
_BINARY = re.compile("[\\w).%π" + _CLOSERS + "]\\s*[+\\-*/×·÷=<>≤≥≠≈±]")
_NESTED = re.compile("[\ue000\ue002\ue004\ue006][^\ue000-\ue007]*[\ue001\ue003\ue005\ue007]")
_INT = re.compile(r"\s*-?\s*\d+\s*")
_NO_OPERAND_BEFORE = "(?<![\\w.,/^" + _CLOSERS + "])"
_MIXED = re.compile(_NO_OPERAND_BEFORE + r"(\d+)[ \t]+(\d+)/(\d+)(?![\d/^]|\.\d)")
_FRACTION = re.compile(_NO_OPERAND_BEFORE + r"(\d+)/(\d+)(?![\d/^]|\.\d)")
_EXPONENT = re.compile(
    "(" + RQ + r")?[ \t]*\^[ \t]*(?:(-?)[ \t]*(\d+)(?!\d|\.\d)|(-?)[ \t]*([a-zA-Z])(?![a-zA-Z])|(" + LE + "))?"
)
_GROUP_AFTER_OPERAND = re.compile(
    "(?<=[\\d" + _CLOSERS + YV + "])(?=[" + LQ + LS + "√∛∜])|(?<![a-zA-Z][a-zA-Z])(?<=[a-zA-Z])(?=[" + LQ + LS + "√∛∜])"
)
_OPERAND_AFTER_GROUP = re.compile("(?<=[" + RQ + RS + "])(?=[\\da-zA-Z" + OS + YV + "])")
_ORDINAL_SIGN = re.compile(r"(?<![\w.,])(10|[1-9])\.?([ºª])")
_ORDINAL_SUFFIX = re.compile(r"(?<![\w.,])(10|[1-9])(er|ero|era|do|da|ro|ra|to|ta|vo|va|no|na|mo|ma)(?![^\W\d_])")
_PERCENT = re.compile(rf"(?<![\d.,])({_NUM})[ \t]*%")
_DEGREES = re.compile(rf"(?<![\d.,])({_NUM})[ \t]*°(?:[ \t]*([CF])(?![a-zA-Z]))?")
_OPERATOR = re.compile("->|=>|[-+=<>≤≥≠≈±×÷/*·→⇒]")
_NUMBER = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_NEXT_WORD = re.compile(r"[ \t]+([^\W\d_]{2,})")
_LETTER_RUN = re.compile(r"(?<![^\W\d_])[a-z]{2,3}(?![^\W\d_])")
_VARIABLE_Y = re.compile(r"(?<![^\W\d_])y(?![^\W\d_])")
_PREV_WORD = re.compile(r"([^\W\d_]+)[ \t]+$")
_Y_CUES = frozenset(["despeja", "despejar", "despejamos", "despejas", "eje", "variable", "valor", "para", "de"])
# fmt: off
_KEEP_RUNS = _NOT_NOUNS | frozenset([
    "cm", "mm", "km", "kg", "mg", "ml", "cl", "dl", "hr", "hrs", "min", "seg", "lb", "lbs", "oz", "ft", "mi", "gal",
    "mph", "am", "pm", "un", "una", "uno", "mi", "tu", "su", "ya", "dos", "mil", "par", "muy", "hay", "ser", "fue",
    "van", "ver", "hoy", "ese", "esa", "eso",
])
# fmt: on
_SOFT_RUN = re.compile("[ \t]*" + SOFT + "[ \t" + SOFT + "]*")


def has_math_symbols(text: str) -> bool:
    return bool(_MATH_HINT.search(text or ""))


def number_to_words_es(n: int, *, fem: bool = False, apocope: bool = False) -> str:
    """Cardinal in words; `apocope` for a following masculine noun ("un", "veintiún"), `fem` for a feminine one."""
    if n < 0:
        return "menos " + number_to_words_es(-n, fem=fem, apocope=apocope)
    if n == 0:
        return "cero"
    if n >= 10**12:
        return " ".join(_UNITS[int(d)] for d in str(n))
    millions, rest = divmod(n, 10**6)
    thousands, units = divmod(rest, 1000)
    parts = []
    if millions:
        parts.append("un millón" if millions == 1 else number_to_words_es(millions, apocope=True) + " millones")
    if thousands:
        parts.append("mil" if thousands == 1 else _below_thousand(thousands, fem, True) + " mil")
    if units:
        parts.append(_below_thousand(units, fem, apocope))
    return " ".join(parts)


def fraction_to_words_es(num: int, den: int) -> str:
    name = _denominator(den)
    if name is None:
        return f"{number_to_words_es(num)} sobre {number_to_words_es(den)}"
    return f"{number_to_words_es(num, apocope=True)} {name if num == 1 else name + 's'}"


def to_speech_es(text: str) -> str:
    if not text or not _MAYBE_MATH.search(text):
        return text
    s = _normalize(text)
    s = _ORDINAL_SIGN.sub(_ordinal, s)
    s = _ORDINAL_SUFFIX.sub(_ordinal, s)
    s = _POINT.sub(_point, s)
    s = _mark_groups(s)
    s = _LETTER_RUN.sub(_split_letters, s)
    s = _VARIABLE_Y.sub(_variable_y, s)
    s = _MIXED.sub(_mixed, s)
    s = _FRACTION.sub(lambda m: _chunk(fraction_to_words_es(int(m[1]), int(m[2]))), s)
    s = _EXPONENT.sub(_exponent, s)
    s = s.replace("π", _chunk("pi")).replace("∞", _chunk("infinito"))
    s = _GROUP_AFTER_OPERAND.sub(TIMES, s)
    s = _OPERAND_AFTER_GROUP.sub(TIMES, s)
    for sign, words in _ROOTS.items():
        s = s.replace(sign, OS + SOFT + words + SOFT)
    s = _PERCENT.sub(lambda m: _chunk(_spell(m[1]) + " por ciento"), s)
    s = s.replace("%", _chunk("por ciento"))
    s = _DEGREES.sub(_degrees, s)
    s = _OPERATOR.sub(_operator, s)
    s = _NUMBER.sub(_number, s)
    return _finish(s)


def _below_thousand(n: int, fem: bool, apocope: bool) -> str:
    if n == 100:
        return "cien"
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        word = _HUNDREDS[hundreds]
        parts.append(word[:-2] + "as" if fem and hundreds > 1 else word)
    if rest:
        tens, unit = divmod(rest, 10)
        if rest < 30:
            parts.append(_one(_UNITS[rest], fem, apocope) if rest in (1, 21) else _UNITS[rest])
        else:
            parts.append(
                _TENS[tens] + (" y " + (_one("uno", fem, apocope) if unit == 1 else _UNITS[unit]) if unit else "")
            )
    return " ".join(parts)


def _one(word: str, fem: bool, apocope: bool) -> str:
    if fem:
        return word[:-1] + "a"
    if apocope:
        return "veintiún" if word == "veintiuno" else "un"
    return word


def _denominator(den: int) -> str | None:
    if den in _DENOMS:
        return _DENOMS[den]
    if 11 <= den <= 29:
        return _UNITS[den].translate(_UNACCENT) + "avo"
    if 30 <= den <= 90 and den % 10 == 0:
        return _TENS[den // 10][:-1] + "avo"
    return None


def _spell(raw: str, *, fem: bool = False, apocope: bool = False) -> str:
    whole, _, frac = raw.replace(",", "").partition(".")
    words = number_to_words_es(int(whole), fem=fem, apocope=apocope and not frac)
    if frac:
        # "3.14" -> "tres punto catorce" like a Mexican classroom; leading zeros / long tails go digit by digit.
        tail = number_to_words_es(int(frac)) if len(frac) <= 2 and not frac.startswith("0") else _digits(frac)
        words += " punto " + tail
    return words


def _digits(digits: str) -> str:
    return " ".join(_UNITS[int(d)] for d in digits)


def _chunk(words: str) -> str:
    return OS + SOFT + words + SOFT + OE


def _normalize(text: str) -> str:
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    text = _UNICODE_FRACTION.sub(lambda m: (m[1] + " " if m[1] else "") + _UNICODE_FRACTIONS[m[2]], text)
    return _SUPERSCRIPT_RUN.sub(lambda m: "^" + m[0].translate(_SUPERSCRIPTS), text)


def _point(m: re.Match[str]) -> str:
    pre = m["pre"] or ""
    label = "" if pre else "el punto"
    return f"{pre}{OS}{SOFT}{label}{SOFT}{_coordinate(m['a'])}{SOFT}, {_coordinate(m['b'])}{SOFT}{OE}"


def _coordinate(raw: str) -> str:
    return re.sub(r"\s+", "", raw).replace("y", YV)


def _prev_char(s: str, i: int) -> str:
    k = i - 1
    while k >= 0 and s[k] in " \t" + SOFT:
        k -= 1
    return s[k] if k >= 0 else ""


def _next_char(s: str, j: int) -> str:
    while j < len(s) and s[j] in " \t" + SOFT:
        j += 1
    return s[j] if j < len(s) else ""


def _split_letters(m: re.Match[str]) -> str:
    # "2ab" / "y = mx + b": a short letter run inside math is a product of variables, not a word.
    run, s = m[0], m.string
    if run in _KEEP_RUNS:
        return run
    prev, nxt = _prev_char(s, m.start()), _next_char(s, m.end())
    coefficient = m.start() > 0 and s[m.start() - 1].isdigit()
    framed = bool(prev) and prev in _OPS + _OPENERS and (not nxt or nxt in _OPS + _CLOSERS + ".,;:?!")
    if not (coefficient or framed):
        return run
    return SOFT + SOFT.join(YV if c == "y" else c for c in run) + SOFT


def _variable_y(m: re.Match[str]) -> str:
    s, i = m.string, m.start()
    prev, nxt = _prev_char(s, i), _next_char(s, m.end())
    if (i > 0 and s[i - 1].isdigit()) or (prev and prev in _OPS + _OPENERS):
        return YV
    # "y - 3 = 0" is the variable, but "son 3 y -2" is the conjunction before a negative number.
    if nxt and (nxt in _OPS.replace("-", "") + _CLOSERS or (nxt == "-" and not prev)):
        return YV
    # "despeja y", "el eje y", "el valor de y": no conjunction can follow these words.
    word = _PREV_WORD.search(s, max(0, i - 24), i)
    return YV if word and word[1].lower() in _Y_CUES else m[0]


def _is_math(inner: str) -> bool:
    if not re.search(r"[\da-zA-Zπ]", inner) or _MATH_CONTENT.fullmatch(inner) is None:
        return False
    # Letter runs are variables only when single, or a short product behind a coefficient ("2ab").
    return all(
        len(run[0]) == 1 or (len(run[0]) <= 3 and run.start() > 0 and inner[run.start() - 1].isdigit())
        for run in re.finditer("[a-zA-Z]+", inner)
    )


def _mark_groups(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        inner = m[1]
        if not _is_math(inner):
            return PL + inner + PR
        before = m.string[: m.start()]
        if before.rstrip(" \t").endswith("^"):
            return re.sub(r"\s+", "", inner) if _INT.fullmatch(inner) else LE + inner + RE
        if before[-1:] in ("f", "g", "h") and not before[-2:-1].isalpha():
            return LF + inner + RF
        top, nested = inner, 1
        while nested:
            top, nested = _NESTED.subn("1", top)
        if _TERM.fullmatch(top) or not _BINARY.search(top):
            return LS + inner + RS
        return LQ + inner + RQ

    replaced = 1
    while replaced:
        s, replaced = _GROUP.subn(repl, s)
    return s.replace(PL, "(").replace(PR, ")")


def _mixed(m: re.Match[str]) -> str:
    whole, num, den = int(m[1]), int(m[2]), int(m[3])
    if not 0 < num < den or _denominator(den) is None:
        return m[0]
    return _chunk(f"{number_to_words_es(whole)} y {fraction_to_words_es(num, den)}")


def _exponent(m: re.Match[str]) -> str:
    # "(x+1)^2" reads "la cantidad x más uno, al cuadrado": the comma keeps the square off the last term.
    lead = RQ + ", " if m[1] else ""
    if m[3] is not None:
        n = int(m[3])
        if m[2]:
            words = "a la menos " + number_to_words_es(n)
        elif n in (2, 3):
            words = "al cuadrado" if n == 2 else "al cubo"
        else:
            words = "a la " + _POWER_ORDINALS.get(n, number_to_words_es(n))
    elif m[5] is not None:
        words = "a la " + ("menos " if m[4] else "") + m[5]
    elif m[6]:
        return lead + SOFT + "a la potencia" + SOFT + LE
    else:
        words = "a la"
    return lead + _chunk(words)


def _ordinal(m: re.Match[str]) -> str:
    masc, fem, short, suffixes = _ORDINALS[int(m[1])]
    mark = m[2]
    if mark in ("º", "ª"):
        return _chunk(fem if mark == "ª" else masc)
    if mark not in suffixes:
        return m[0]
    return _chunk(short if mark == "er" else fem if mark.endswith("a") else masc)


def _degrees(m: re.Match[str]) -> str:
    unit = "un grado" if m[1] == "1" else _spell(m[1], apocope=True) + " grados"
    scale = {"C": " Celsius", "F": " Fahrenheit"}.get(m[2] or "", "")
    return _chunk(unit + scale)


def _ends_operand(s: str, i: int) -> bool:
    k = i - 1
    while k >= 0 and s[k] in " \t" + SOFT:
        k -= 1
    if k < 0:
        return False
    c = s[k]
    if c.isascii() and c.isalpha():
        return k == 0 or not s[k - 1].isalpha()
    return c.isdigit() or c in ")" + _CLOSERS + YV


def _starts_operand(s: str, j: int, *, skip: bool) -> bool:
    k = j
    while skip and k < len(s) and s[k] in " \t" + SOFT:
        k += 1
    if k >= len(s):
        return False
    c = s[k]
    if c.isascii() and c.isalpha():
        return k + 1 >= len(s) or not s[k + 1].isalpha()
    return c.isdigit() or c in "(" + LQ + LS + OS + YV


def _operator(m: re.Match[str]) -> str:
    op, s, i, j = m[0], m.string, m.start(), m.end()
    if op == "-":
        # A hyphen only reads "menos" next to an operand: keeps "auto-evaluación" and "- paso" bullets as prose.
        if not (_starts_operand(s, j, skip=True) and (_ends_operand(s, i) or _starts_operand(s, j, skip=False))):
            return op
        words = "menos"
    elif op in ("*", "·"):
        # Outside an a*b product an asterisk is markdown emphasis: drop it rather than read "por".
        if not (_ends_operand(s, i) and _starts_operand(s, j, skip=True)):
            return ""
        words = "por"
    elif op == "/":
        words = "sobre" if _ends_operand(s, i) or _starts_operand(s, j, skip=True) else "o"
    else:
        words = _OPERATOR_WORDS[op]
    return SOFT + words + SOFT


def _noun_gender(word: str) -> str | None:
    lower = word.lower()
    if word[0].isupper() or lower in _NOT_NOUNS:
        return None
    base = lower[:-1] if lower.endswith("s") else lower
    if lower in _FEM_OTHER or base in _FEM_OTHER:
        return "f"
    if base in _MASC_A:
        return "m"
    return "f" if base.endswith(("a", "ión", "dad", "tad", "tud")) or lower.endswith("iones") else "m"


def _number(m: re.Match[str]) -> str:
    fem = apocope = False
    if m[2] is None:
        nxt = _NEXT_WORD.match(m.string, m.end())
        gender = _noun_gender(nxt[1]) if nxt else None
        fem, apocope = gender == "f", gender == "m"
    return SOFT + _spell(m[0], fem=fem, apocope=apocope) + SOFT


def _finish(s: str) -> str:
    s = (
        s.replace(YV, "ye")
        .replace(LQ, SOFT + "la cantidad" + SOFT)
        .replace(LF, SOFT + "de" + SOFT)
        .replace(TIMES, SOFT + "por" + SOFT)
    )
    s = re.sub("[\ue000-\ue00d]", SOFT, s)

    def collapse(m: re.Match[str]) -> str:
        if " " in m[0] or "\t" in m[0]:
            return " "
        before = m.string[m.start() - 1] if m.start() else ""
        after = m.string[m.end()] if m.end() < len(m.string) else ""
        return " " if before.isalnum() and after.isalnum() else ""

    return _SOFT_RUN.sub(collapse, s)
