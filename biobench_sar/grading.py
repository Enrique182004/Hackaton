"""Quiz answers graded in code, and the solution of a board problem worked out in code (sympy).

parse_answers reads an answer as written or said ("x = 5", "5/2", "x = 5 or x = -3", "x = 5, y = 2", "(2, 3)",
"x es igual a cinco", one answer per line) into a set of (letter, value) pairs, and grade compares a student's answer
with the expected one: the model only copies the answer off the board, it never decides whether it is right.
Arithmetic not worked out yet ("1/2 + 1/3" copied off the board, "2/6 + 3/6") is no answer (unfinished), and a check
or a reason written after an answer ("x = 5 (check: 3(5) + 5 = 20)", "because ...") is cut off. solve_text solves the
problem itself (linear, quadratic, a system), so a help reply can be checked for stating the solution
(solution_sentences) and a wrong quiz answer can get one hint for the next step (hint_for).

Nothing here raises on bad input: unreadable text gives an empty set or None. The parser is our own (no eval), so a
model's reply or an uploaded file can never run code. Nothing here takes long either (it runs on the assistant
thread, and a big-int power holds the GIL): the size of a power or a root and the degree of an equation are bounded
before they are worked out (MAX_BITS, MAX_POWER_DEGREE, MAX_DEGREE), and only what solves fast is solved (systems
with at most one quadratic, one letter up to a quadratic, what factors into them or a power "x^3 = 2"). Without
sympy, plain numbers and one-letter linear problems still work (quiz_bank.solve_problem); square roots, quadratics
and systems then give None.
"""

from __future__ import annotations

import functools
import logging
import math
import re
import unicodedata
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

log = logging.getLogger(__name__)

Value = Any  # a sympy number, or a Fraction when sympy is missing
Answer = tuple[str, Value]  # (letter, value): "" for a bare value, "#1" / "#2" for the entries of a pair "(2, 3)"
Answers = frozenset[Answer]

MAX_CHARS = 240
MAX_EXPONENT = 6
MAX_LETTERS = 3
MAX_DEGREE = 4
# What a power or a root may come to, checked before it is worked out: a 37-character tower "((((9)^6)^6)^6)^6..."
# is 9^(6^8), 1.6 million digits that a big-int pow works out holding the GIL (camera and sockets frozen for
# seconds). 1024 bits is about 300 digits, far past any school answer, and still instant.
MAX_BITS = 1024
MAX_POWER_DEGREE = 2 * MAX_EXPONENT
TOLERANCE = 1e-9
_NONE: Answers = frozenset()
# What a quiz problem is, for telling a check from the answer (_problem_kind).
_Kind = tuple[bool, Any] | None


@functools.lru_cache(maxsize=1)
def _sympy() -> Any:
    try:
        import sympy
    except Exception:  # missing or broken: plain numbers still work
        log.warning("sympy is not available: only plain numbers and linear problems are graded")
        return None
    return sympy


# ---- text -> plain math ----------------------------------------------------------------------------------------

_SUPERSCRIPTS = str.maketrans({"²": "^2", "³": "^3", "⁴": "^4"})
_CHARS = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "×": "*",
        "·": "*",
        "∙": "*",
        "⋅": "*",
        "⁄": "/",
        "：": ":",
        "＝": "=",
        "，": ",",
    }
)
# One arrow for all: "->" is no minus sign and "=>" no "=" ("3(5) + 5 = 20 ->", "2x + 3 = 11 => x = 4").
_ARROWS = str.maketrans({"⇒": "→", "⟹": "→", "⟶": "→", "➔": "→", "⇨": "→"})
_BOLD = re.compile(r"(?<=[\w)])\*\*(?=[\w(])")

_EN_SMALL = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ],
        range(20),
        strict=True,
    )
)
_EN_TENS = dict(
    zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10), strict=True)
)
_ES_SMALL = dict(
    zip(
        [
            "cero",
            "uno",
            "dos",
            "tres",
            "cuatro",
            "cinco",
            "seis",
            "siete",
            "ocho",
            "nueve",
            "diez",
            "once",
            "doce",
            "trece",
            "catorce",
            "quince",
            "dieciseis",
            "diecisiete",
            "dieciocho",
            "diecinueve",
            "veinte",
            "veintiuno",
            "veintidos",
            "veintitres",
            "veinticuatro",
            "veinticinco",
            "veintiseis",
            "veintisiete",
            "veintiocho",
            "veintinueve",
        ],
        range(30),
        strict=True,
    )
)
_ES_SMALL |= {"un": 1, "una": 1, "veintiun": 21, "veintiuna": 21}
# "un" / "una" are numbers only before a fraction word ("un medio"): "x es una variable" says nothing about 1.
_ES_ARTICLE_ONE = frozenset({"un", "una"})
_ES_TENS = dict(
    zip(
        ["treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"],
        range(30, 100, 10),
        strict=True,
    )
)
_ES_HUNDREDS = {"cien": 100, "ciento": 100, "quinientos": 500, "quinientas": 500, "setecientos": 700}
_ES_HUNDREDS |= {"novecientos": 900, "novecientas": 900, "setecientas": 700}
_ES_HUNDREDS |= {f"{p}cientos": n * 100 for p, n in (("dos", 2), ("tres", 3), ("cuatro", 4), ("seis", 6), ("ocho", 8))}
_ES_HUNDREDS |= {f"{p}cientas": n * 100 for p, n in (("dos", 2), ("tres", 3), ("cuatro", 4), ("seis", 6), ("ocho", 8))}
_EN_FRACTIONS = {"half": 2, "halves": 2, "quarter": 4, "quarters": 4}
_EN_FRACTIONS |= {
    w + s: n
    for n, w in enumerate(
        ["third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"], 3
    )
    for s in ("", "s")
}
_ES_FRACTIONS = {
    w + s: n
    for n, w in enumerate(["medio", "tercio", "cuarto", "quinto", "sexto", "septimo", "octavo", "noveno", "decimo"], 2)
    for s in ("", "s")
}
_ES_FRACTIONS |= {"media": 2}
_TABLES = {
    "en": (_EN_SMALL, _EN_TENS, {}, _EN_FRACTIONS, {"thousand"}, {"hundred"}),
    "es": (_ES_SMALL, _ES_TENS, _ES_HUNDREDS, _ES_FRACTIONS, {"mil"}, set()),
}
_SIGN_WORDS = {"en": {"negative", "minus"}, "es": {"menos", "negativo", "negativa"}}
_POINT_WORDS = {"en": {"point"}, "es": {"punto", "coma"}}
_OPERATOR_WORDS = {
    "en": {"plus": "+", "times": "*", "over": "/"},
    "es": {"mas": "+", "por": "*", "entre": "/", "sobre": "/"},
}
_PHRASES = {
    "en": (
        (r"\bdivided\s+by\b", " / "),
        (r"\bmultiplied\s+by\b", " * "),
        (r"\bthe\s+square\s+root\s+of\b|\bsquare\s+root\s+of\b", " √ "),
        (r"\bsquared\b", " ^2 "),
        (r"\bcubed\b", " ^3 "),
        (r"\bplus\s+or\s+minus\b", " ± "),
        (
            r"\b(?:is\s+equal\s+to|is\s+equals|equals|equal\s+to|would\s+be|will\s+be|should\s+be|must\s+be|"
            r"comes\s+out\s+to|is)\b",
            " = ",
        ),
    ),
    "es": (
        (r"\bdividid[oa]s?\s+(?:entre|por|sobre)\b", " / "),
        (r"\bmultiplicad[oa]s?\s+por\b", " * "),
        (r"\b(?:la\s+)?raiz\s+cuadrada\s+de\b", " √ "),
        (r"\bal\s+cuadrado\b", " ^2 "),
        (r"\bal\s+cubo\b", " ^3 "),
        (r"\bmas\s+o\s+menos\b", " ± "),
        (r"\bpor\s+(?:favor|eso|ejemplo|lo\s+tanto|fin|que|cierto|ahora|cada|ambos|los\s+dos|un\s+lado)\b", " "),
        (
            r"\b(?:es\s+igual\s+a|son\s+iguales\s+a|igual\s+a|equivale\s+a|deberia\s+ser|debe\s+ser|seria|sera|"
            r"vale|valen|queda\s+en|nos\s+da|te\s+da|resulta\s+en|es)\b",
            " = ",
        ),
        (r"\bye\b", " y "),
    ),
}
_WORD_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?|\.\d+|\s+|.")
# "2,5" is two and a half (Spanish); "(2,3)" and "1,2,3" are lists.
_DECIMAL_COMMA = re.compile(r"(?<![\d(,])(\d+),(\d+)(?![\d,)])")


def _plain(text: Any) -> str:
    """Lowercase, accents off, one symbol per operation ("−" -> "-", "×" -> "*", "x²" -> "x^2", "：" -> ":")."""
    raw = str(text if text is not None else "")[: MAX_CHARS * 4].translate(_SUPERSCRIPTS)
    raw = unicodedata.normalize("NFKC", raw).translate(_CHARS).lower()
    raw = "".join(c for c in unicodedata.normalize("NFD", raw) if not unicodedata.combining(c))
    raw = _BOLD.sub("^", raw).replace("**", "").replace("`", "").replace("$", "")
    raw = _DECIMAL_COMMA.sub(r"\1.\2", raw)
    raw = raw.translate(_ARROWS).replace("-->", "→").replace("->", "→").replace("=>", "→")
    return raw.replace("+/-", "±").replace("+-", "±")


def _languages(language: str | None) -> tuple[str, ...]:
    if language in ("en", "es"):
        return (language,)
    return ("en", "es")


class _Words:
    """Number words to digits for one language: "twenty five" -> 25, "dos tercios" -> 2/3, "two point five" -> 2.5,
    "treinta y cinco" -> 35, "negative four" / "menos cuatro" -> -4."""

    def __init__(self, language: str) -> None:
        small, tens, hundreds, fractions, thousands, multipliers = _TABLES[language]
        self.language = language
        self.small, self.tens, self.hundreds, self.fractions = small, tens, hundreds, fractions
        self.thousands, self.multipliers = thousands, multipliers
        self.signs, self.points = _SIGN_WORDS[language], _POINT_WORDS[language]
        self.operators = _OPERATOR_WORDS[language]

    def run(self, words: list[str], i: int) -> tuple[int, int] | None:
        """The number spelled by words[i:], and the index after it; None when words[i] starts no number."""
        total, current, last, j = 0, 0, None, i
        while j < len(words):
            word = words[j]
            if word in self.small and last in (None, "hundred", "thousand", "tens"):
                if last == "tens" and (self.small[word] >= 10 or self.language == "es"):
                    break  # "twenty twelve"; Spanish joins tens and units with "y" ("treinta y cinco")
                if word in _ES_ARTICLE_ONE and self.language == "es" and last is None:
                    nxt = words[j + 1] if j + 1 < len(words) else ""
                    if nxt not in self.fractions and nxt not in self.thousands:
                        break
                current, last = current + self.small[word], "small"
            elif word in self.tens and last in (None, "hundred", "thousand"):
                current, last = current + self.tens[word], "tens"
            elif word in self.hundreds and last in (None, "thousand"):
                current, last = current + self.hundreds[word], "hundred"
            elif word in self.multipliers and last in ("small", "tens"):
                current, last = current * 100, "hundred"
            elif word in self.thousands and last in (None, "small", "tens", "hundred"):
                total, current, last = total + max(current, 1) * 1000, 0, "thousand"
            elif word == "y" and self.language == "es" and last == "tens":
                nxt = words[j + 1] if j + 1 < len(words) else ""
                if nxt not in self.small or self.small[nxt] >= 10 or nxt in _ES_ARTICLE_ONE:
                    break
                j += 1
                current, last = current + self.small[nxt], "small"
            elif word == "and" and self.language == "en" and last in ("hundred", "thousand"):
                nxt = words[j + 1] if j + 1 < len(words) else ""
                if not (nxt in self.small or nxt in self.tens):
                    break
            else:
                break
            j += 1
        if last is None:
            return None
        return total + current, j


def _words_to_math(text: str, language: str) -> str:
    words_of = _Words(language)
    if language == "en":  # "twenty-five", "five-sixths": one number, not 20 - 5
        text = _HYPHENATED.sub(lambda m: f"{m[1]} {m[2]}" if _number_word(m[1], m[2], words_of) else m[0], text)
    for pattern, replacement in _PHRASES[language]:
        text = re.sub(pattern, replacement, text)
    tokens = _WORD_TOKEN.findall(text)
    # Number runs skip the spaces between their words; everything else keeps its spacing ("2y" stays one term).
    at = [k for k, t in enumerate(tokens) if t.strip()]
    words = [tokens[k] for k in at]
    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        found = words_of.run(words, i) if word.isalpha() else None
        if word == "a" and language == "en" and i + 1 < len(words) and words[i + 1] in words_of.fractions:
            found = (1, i + 1)
        if found is None:
            out.append(words_of.operators.get(word, "-" if word in words_of.signs else word))
            out.append(_space_after(tokens, at, i))
            i += 1
            continue
        start = i
        value, i = found
        piece = str(value)
        nxt = words[i] if i < len(words) else ""
        if nxt in words_of.fractions:
            piece, i = f"{value}/{words_of.fractions[nxt]}", i + 1
        elif nxt in words_of.points and i + 1 < len(words):
            digits: list[str] = []
            j = i + 1
            while j < len(words) and (step := words_of.run(words, j)) is not None:
                digits.append(str(step[0]))
                j = step[1]
            if digits:
                piece, i = f"{value}.{''.join(digits)}", j
        tail = words[i : i + 3]
        if language == "en" and tail[:3] == ["and", "a", "half"]:
            piece, i = f"{value} 1/2", i + 3
        elif language == "es" and tail[:2] in (["y", "medio"], ["y", "media"]):
            piece, i = f"{value} 1/2", i + 2
        out.append(piece if i > start else word)
        out.append(_space_after(tokens, at, i - 1))
    return "".join(out)


_HYPHENATED = re.compile(r"\b([a-z]+)-([a-z]+)\b")


def _number_word(first: str, second: str, words_of: _Words) -> bool:
    """Whether "first-second" is one number written with a hyphen: "twenty-five", "five-sixths", "one-half"."""
    starts = first in words_of.small or first in words_of.tens or first == "a"
    return starts and (second in words_of.small or second in words_of.fractions)


def _space_after(tokens: list[str], at: list[int], i: int) -> str:
    """The whitespace that followed word i in the text ("" when the next token touches it)."""
    k = at[i] + 1
    return tokens[k] if k < len(tokens) and not tokens[k].strip() else ""


def to_math(text: Any, language: str | None = None) -> str:
    """Text as plain math: number words as digits and "equals" / "is" / "es igual a" / "vale" as "=", in `language`
    ("en" | "es"; None tries English then Spanish words). Other words are kept, and so are line breaks: two lines
    are two answers ("x = 5" over "x = -3")."""
    plain = _plain(text)
    for lang in _languages(language):
        plain = _words_to_math(plain, lang)
    return "\n".join(line for line in (" ".join(part.split()) for part in plain.splitlines()) if line)


# ---- expressions: our own parser, no eval ----------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?|\.\d+)|(?P<sqrt>√|sqrt\b)|(?P<var>[a-z])|(?P<op>[-+*/÷^()]))")


class _Bad(ValueError):
    pass


class _Parser:
    """+ - * / ÷ ^ ( ) √ with implicit products ("2x", "3(x + 1)"); a written fraction bar binds tighter than × and
    ÷, so "5/6 ÷ 1/3" is (5/6) ÷ (1/3) and "1/2x" is half of x. Values are sympy numbers and symbols, or Fractions
    without sympy (then letters and roots are refused)."""

    def __init__(self, text: str, letters: bool) -> None:
        self.tokens = _tokens(text)
        self.pos = 0
        self.letters = letters
        self.sp = _sympy()

    def parse(self) -> Value:
        if not self.tokens:
            raise _Bad("empty")
        value = self.expr()
        if self.pos != len(self.tokens):
            raise _Bad("trailing tokens")
        return value

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise _Bad("unexpected end")
        self.pos += 1
        return token

    def expr(self) -> Value:
        value = self.term()
        while (token := self.peek()) is not None and token[1] in ("+", "-"):
            self.take()
            value = value + self.term() if token[1] == "+" else value - self.term()
        return value

    def term(self) -> Value:
        value = self.fraction()
        while (token := self.peek()) is not None:
            if token[1] in ("*", "÷"):
                self.take()
                value = value * self.fraction() if token[1] == "*" else _divide(value, self.fraction())
            elif token[0] in ("num", "var", "sqrt") or token[1] == "(":
                value = value * self.fraction()
            else:
                break
        return value

    def fraction(self) -> Value:
        value = self.unary()
        while (token := self.peek()) is not None and token[1] == "/":
            self.take()
            value = _divide(value, self.unary())
        return value

    def unary(self) -> Value:
        token = self.peek()
        if token is not None and token[1] in ("+", "-"):
            self.take()
            value = self.unary()
            return -value if token[1] == "-" else value
        return self.power()

    def power(self) -> Value:
        base = self.atom()
        token = self.peek()
        if token is None or token[1] != "^":
            return base
        self.take()
        exponent = self.unary()
        try:
            whole = int(exponent)
        except (TypeError, ValueError) as exc:
            raise _Bad("exponent") from exc
        if whole != exponent or abs(whole) > MAX_EXPONENT:
            raise _Bad("exponent")
        bits, degree = _size(base)
        if bits * abs(whole) > MAX_BITS or degree * abs(whole) > MAX_POWER_DEGREE:
            raise _Bad("too big")  # never worked out: each level of a tower is six times the last one
        if whole < 0:
            return _divide(1, base**-whole)
        return base**whole

    def atom(self) -> Value:
        kind, text = self.take()
        if kind == "num":
            return self.sp.Rational(text) if self.sp is not None else Fraction(text)
        if kind == "var":
            if not self.letters or self.sp is None:
                raise _Bad("letter")
            return self.sp.Symbol(text)
        if kind == "sqrt":
            if self.sp is None:
                raise _Bad("root")
            inside = self.atom()
            if _size(inside)[0] > MAX_BITS:
                raise _Bad("too big")  # sympy looks for square factors of the number
            return self.sp.sqrt(inside)
        if text == "(":
            value = self.expr()
            if self.take()[1] != ")":
                raise _Bad("brackets")
            return value
        raise _Bad(f"unexpected {text!r}")


def _size(value: Value) -> tuple[int, int]:
    """(bits, degree): upper bounds on how big the numbers in `value` get and on its degree once multiplied out,
    read off its tree without working anything out. A root counts as a letter: (√2 + √3)^6 multiplies out too."""
    sp = _sympy()
    if isinstance(value, (Fraction, int)):
        exact = Fraction(value)
        return abs(exact.numerator).bit_length() + exact.denominator.bit_length(), 0
    if sp is None or not isinstance(value, sp.Basic):
        return 0, 0
    if value.is_Rational:
        return abs(int(value.p)).bit_length() + int(value.q).bit_length(), 0
    if value.is_Symbol:
        return 1, 1
    if value.is_Pow:
        bits, degree = _size(value.base)
        if value.exp.is_Integer:
            n = abs(int(value.exp))
            return bits * n, degree * n
        return bits, max(degree, 1)
    sizes = [_size(arg) for arg in value.args]
    if not sizes:
        return 1, 1
    if value.is_Add:
        return max(b for b, _ in sizes) + len(sizes), max(d for _, d in sizes)
    return sum(b for b, _ in sizes), sum(d for _, d in sizes)


def _divide(a: Value, b: Value) -> Value:
    try:
        if b == 0:
            raise _Bad("divide by zero")
    except TypeError:  # a sympy expression with letters: never "== 0" as written
        pass
    return a / b


def _tokens(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos, text = 0, text.strip()
    while pos < len(text):
        found = _TOKEN.match(text, pos)
        if not found:
            if text[pos:].strip():
                raise _Bad(f"not math at {pos}")
            break
        kind = found.lastgroup or "op"
        tokens.append((kind, found.group(kind)))
        pos = found.end()
    return tokens


def _expression(text: str, *, letters: bool = False) -> Value | None:
    try:
        value = _Parser(text, letters).parse()
    except (_Bad, ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None
    sp = _sympy()
    if sp is not None and isinstance(value, sp.Basic) and value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    return value


def _number(value: Value) -> float | None:
    """A real value as a float, else None (letters left, or not real)."""
    try:
        sp = _sympy()
        if sp is not None and isinstance(value, sp.Basic):
            if value.free_symbols:
                return None
            number = complex(value.evalf())
            if abs(number.imag) > TOLERANCE:
                return None
            result = number.real
        else:
            result = float(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


def _close(a: Value, b: Value) -> bool:
    x, y = _number(a), _number(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


_MIXED = re.compile(r"^(-?)\s*(\d+)\s+(\d+)\s*/\s*(\d+)$")
_VALUE_CHARS = re.compile(r"[\d\s.+\-*/÷^()√±%]+")


def _values(text: str) -> list[Value] | None:
    """The value(s) of a written value: "5", "-3", "5/2", "2.5", "2 1/2", "√2", "±3" (two), "75%"; None otherwise."""
    s = text.strip().rstrip(" .,;:!?+-*/÷^(").strip()
    if s.count(")") > s.count("("):
        s = s.rstrip(")").strip()
    both = s.startswith("±")
    s = s.lstrip("±").strip()
    percent = s.endswith("%")
    s = s.rstrip("%").strip()
    if not s or "%" in s or "±" in s:
        return None
    mixed = _MIXED.match(s)
    if mixed:
        sign, whole, num, den = mixed.groups()
        if int(den) == 0:
            return None
        sp = _sympy()
        part = sp.Rational(int(num), int(den)) if sp is not None else Fraction(int(num), int(den))
        value: Value = (-1 if sign else 1) * (int(whole) + part)
    else:
        value = _expression(s)
        if value is None or _number(value) is None:
            return None
    if percent:
        value = value / 100
    return [value, -value] if both else [value]


# ---- answers ---------------------------------------------------------------------------------------------------

_PAIR = re.compile(r"\(\s*([^(),;=]+?)\s*[,;]\s*([^(),;=]+?)\s*\)")  # "(2, 3)"; "(x = 7, y = 3)" holds answers
# "and" / "or" / "o" / "y" / "," between answers; a "y" next to "=" is the letter y ("x = 5, y = 2", "3 = y").
_CLAUSES = re.compile(
    r"\s*(?:,(?!\d)|;|\n|&|\bor\b|\band\b|\bthen\b|\bso\b|\bo\b|\bluego\b|\bentonces\b|(?<!=)(?<!= )\by\b(?!\s*=)"
    r"|\.\s)\s*"
)
_LABEL = re.compile(
    r"^\s*(?:(?:the|my|your|our|final|correct|la|mi|tu|el)\s+)*"
    r"(?:answers?|ans|solutions?|sol|results?|respuestas?|resp|solucion(?:es)?|resultados?)"
    r"\b(?:\s+(?:correcta|final|correct))?\s*(?:[:=.\-]\s*|\s+(?=[-+\d(√±]))?"
    r"|^\s*[ar]\d{0,3}\s*[:.]\s*"
)
# A letter set to something: the letter is not a term of an expression ("3x = 15", "x + 4 = 10" are no answers).
_ASSIGNED = re.compile(r"(?<![\w.])([a-z])\s*=\s*")
# The letter alone on the right of the only "=": "3 = x" (from "7 - 2x = 1": "6 = 2x", then "3 = x"), also with a
# mark after it ("3 = x ✓", "3 = x <-", "3 = x answer"); not "3 = x + 1", "3 = xy", "3 = x 2".
_ASSIGNED_RIGHT = re.compile(r"=\s*([a-z])(?![a-z])(?=\s*$|\s+(?![\d.(+\-*/÷^√±]|[a-z](?![a-z]))[^\d=]*$)")
_TERM_END = re.compile(r"[\d)^*/÷+\-]\s*$")
# A single letter right after a value makes it an expression ("x = 2y + 1", "x = 5x"); "y luego" is a word.
_VARIABLE_NEXT = re.compile(r"\s*[a-z]\s*(?:[=+\-*/^(]|\d|$)")
_LEADING_WORDS = re.compile(r"^(?:(?:[a-z]{2,}|i)\s+)+")
# A reason or a check written after an answer: "x = 4 because 2 times 4 is 8", "x = 6: subtract 9 from both sides,
# then divide by 2", "x = 6 - divide by 2", "x = 5; check it". It runs up to the next answer ("..., y = 3") or the line
# end. Not "3:4" (a ratio) and not "A1: x = 4" or "x = 5; y = 2" (a label, a letter).
_NEXT_ANSWER = r"(?=(?:[,;\n]|\b(?:and|or|then|so|y|o|luego|entonces)\b)\s*[a-z]\s*=|\n|$)"
_EXPLANATION = re.compile(
    r"\s*(?:\b(?:because|since|cause|check(?:ing)?|proof|porque|pues|ya\s+que|dado\s+que|comprobacion|"
    r"comprobando|comprueba|verificacion|verificando|verifica)\b(?!\s*:?\s*[a-z]\s*=)"
    r"|(?:(?<=\d)(?<![a-z]\d)(?<![a-z]\d\d)|(?<=\)))\s*(?::|;|\s-)\s+(?=[a-z]{2,}))"
    rf".*?{_NEXT_ANSWER}"
)
# Brackets that open on a word hold a check or a note: "x = 5 (check: 3(5) + 5 = 20)", "x = 6 (subtract nine)".
_ASIDE = re.compile(r"[(\[]\s*(?!sqrt\b)(?=[a-z]{2,})")
# "x = 5 x = -3", "x = 7 y = 3": answers next to each other with only a space between them; "3 = x 7 - 2(3) = 1" and
# "x = 5 3(5) + 5 = 20": an answer and its check (not "x = 2 1/2 = 5/2", "x = 15 - 5 = 10", "x = 2 (3 + 1) = 8").
_GAP = re.compile(
    r"(=[^=;,\n]*?[\d)])\s+(?=[a-z]\s*=)|(=\s*[a-z])\s+(?=[\d√][^=;,\n]*=)"
    r"|(=\s*-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s+(?=[\d√][^=;,\n]*?[-+*÷(][^=;,\n]*=)"
)
# A line that starts with "=" goes on from the one above it: "1/2 + 1/3" over "= 3/6 + 2/6" over "= 5/6".
_GOES_ON = re.compile(r"\s*[;\n]\s*(?==)")
# Subscripts of a root: "x₁ = 3, x₂ = 4" (and "x_1", or "x1 = 3" as a model copies it) are two values of x.
_SUBSCRIPT = re.compile(r"(?<=[A-Za-z])(?:[₀-₉]+|_\d+)|(?<![\w.][A-Za-z])(?<=[A-Za-z])[1-3](?=\s*=)")


def _value_run(text: str) -> tuple[str, str]:
    """(the math at the start of `text`, what follows it)."""
    found = _VALUE_CHARS.match(text)
    if not found:
        return "", text
    return found.group(), text[found.end() :]


def _letter_next(run: str, after: str) -> bool:
    """Whether a letter after a value run makes it a term: "2y + 1", "5x", and a letter touching it whatever follows
    ("6 = 2x ✓", "x = 2y ok"). "5 y luego" (a word after a space) and "5cm" (a unit) are no term."""
    touching = after[:1].isalpha() and not after[1:2].isalpha() and not run[-1:].isspace()
    return touching or bool(_VARIABLE_NEXT.match(after))


def _result_run(side: str) -> tuple[str, str]:
    """(the value a line comes to, at the start of its last side; what follows it). A mark after it is no math:
    "20 ✓", "20 ok", "20 :)", "20 →", and "20 √", a tick read as a root sign."""
    run, after = _value_run(side.strip())
    return re.sub(r"√\s*$", "", run), after


def _bracket_end(text: str, start: int) -> int:
    """Where the bracket opened at `start` closes (after it); an unclosed bracket runs to the end."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] in "([":
            depth += 1
        elif text[i] in ")]":
            depth -= 1
        if depth == 0:
            return i + 1
    return len(text)


def _without_asides(text: str) -> str:
    while (found := _ASIDE.search(text)) is not None:
        text = f"{text[: found.start()]} ; {text[_bracket_end(text, found.start()) :]}"
    # A check of plain math in brackets after an answer: "x = 5 (3(5) + 5 = 20)", not "(x = 7, y = 3)".
    start = 0
    while (opened := text.find("(", start)) >= 0:
        end = _bracket_end(text, opened)
        inside = text[opened + 1 : end - 1]
        if "=" in inside and not re.search(r"[a-z]", inside) and re.search(r"[\d)]\s*$", text[:opened]):
            text = f"{text[:opened]} ; {text[end:]}"
            start = opened
        else:
            start = opened + 1
    return text


def _answer_text(text: Any) -> str:
    """`text` as plain math with what is no answer cut off: checks and reasons ("x = 5 (check: ...)", "because ...",
    ": subtract 9 ..."), and a clause break between answers on their own lines or side by side ("x = 5 x = -3"). A
    line that starts with "=" goes on from the line above it."""
    text = _SUBSCRIPT.sub("", text) if isinstance(text, str) else text
    chained = re.sub(r"^\s*=\s*", "", _GOES_ON.sub(" ", to_math(text)))  # "= 5/6" alone: the bottom of a chain
    math_text = _EXPLANATION.sub(" ; ", _without_asides(chained))
    return _GAP.sub(lambda found: f"{next(g for g in found.groups() if g)} ; ", math_text)


def _assignment(clause: str) -> tuple[str, list[Value]] | None:
    """("x", [5]) for "x = 5" / "so x = 15/3 = 5" / "i got x = 5 apples"; None when no lone letter is set to a value."""
    for found in _ASSIGNED.finditer(clause):
        before = clause[: found.start()]
        if _TERM_END.search(before):
            continue
        run, after = _value_run(clause[found.end() :].strip())
        while after.startswith("="):  # "x = 15/3 = 5": the last value of the chain, never one after words
            run, after = _value_run(after[1:].strip())
        if not run.strip() or _letter_next(run, after):
            return None  # "x = 2y", "x = 5x": not a value
        values = _values(run)
        return (found.group(1), values) if values else None
    return None


def _right_assignment(clause: str) -> tuple[str, list[Value]] | None:
    """("x", [3]) for "3 = x" / "i got 3 = x": a value, then its letter alone on the right; None otherwise."""
    found = _ASSIGNED_RIGHT.search(clause)
    if found is None or clause.count("=") != 1:
        return None
    run, after = _result_run(_LEADING_WORDS.sub("", clause[: found.start()]))
    values = _values(run) if run.strip() and not after.strip() else None
    return (found.group(1), values) if values else None


def _rational(value: Value) -> bool:
    sp = _sympy()
    if sp is not None and isinstance(value, sp.Basic):
        return bool(value.is_Rational)
    return isinstance(value, (Fraction, int))


def _unworked(run: str) -> bool:
    """Operations still to be done ("1/2 + 1/3", "2/6 + 3/6", "12 ÷ 4", "2^3", "1/2 +"): they come to one plain
    number, so it is not the answer yet. One value ("5/6", "-3", "1 1/2", "2.5") or an exact root ("1 + √2") is."""
    s = run.strip().rstrip(" .,;:!?")
    try:
        tokens = _tokens(s)
    except _Bad:
        return False
    if len(tokens) > 1 and tokens[-1][1] in ("+", "-", "*", "/", "÷", "^", "("):
        return True

    def after_term(k: int) -> bool:
        return k > 0 and (tokens[k - 1][0] == "num" or tokens[k - 1][1] == ")")

    operations = [k for k, (_, t) in enumerate(tokens) if t in ("+", "-", "*", "÷", "^", "(") and after_term(k)]
    if not operations and sum(t == "/" for _, t in tokens) < 2:
        return False
    value = _expression(s)
    return value is not None and _rational(value)


def _worked_value(clause: str) -> list[Value] | None:
    """[5/6] for "1/2 + 1/3 = 5/6" (arithmetic and the value it comes to); [] when it is not worked out to one value
    ("1/2 + 1/3 = 2/6 + 3/6", "12 ÷ 4 = ?"); None when it is no plain arithmetic (a letter in it, on either side: a
    problem, a step or "3 = x")."""
    *steps, last = _LEADING_WORDS.sub("", clause).split("=")
    if not steps or any(_expression(s.strip()) is None for s in steps):
        return None
    run, after = _result_run(last)
    if _letter_next(run, after):
        return None  # the letter on the right: "6 = 2x" is a step, "3 = x" an answer (_right_assignment)
    if not run.strip() or _unworked(run):
        return []
    return _values(run) or []


def _problem_kind(problem: Any, expected: Answers | None = None) -> _Kind:
    """(True, None) for a problem with a letter ("3x + 5 = 20", also with words around it: "Despeja x: 3x + 5 = 20",
    "2x - 7 = 9. Find x."), (False, its value) for arithmetic ("1/2 + 1/3"). For a problem code can't read (a word
    problem), the `expected` answer tells: (True, None) when it has a letter ("x = 5"), (False, None) when it is a
    bare value ("12"). None without a problem, or when neither tells."""
    if problem is None:
        return None
    parts = _equations(problem)
    if parts and any(re.search(r"[a-z]", part.replace("sqrt", "")) for part in parts):
        return True, None
    value = _expression(parts[0]) if parts and len(parts) == 1 and "=" not in parts[0] else None
    if value is not None and _number(value) is not None:
        return False, value
    if parts is None and _worded_equation(problem):
        return True, None
    if expected:
        return any(letter for letter, _ in expected), None
    return None


def _worded_equation(problem: Any) -> bool:
    """Whether a problem written with words holds an equation with a letter: "Despeja x: 3x + 5 = 20", "2x - 7 = 9.
    Find x.", "Halla x en 3x + 5 = 20", "Resuelve la ecuación 7 - 2x = 1", "Solve for x in the equation 2x + 3 = 11"."""
    text = to_math(_INSTRUCTION.sub("", _plain(problem).strip()))
    for piece in _SENTENCE_BREAK.split(text):
        math_text = _WORDS_AFTER.sub("", _WORDS_BEFORE.sub("", piece.strip())).strip()
        parts = (_equations(math_text) if "=" in math_text else None) or []
        lettered = any(re.search(r"[a-z]", part.replace("sqrt", "")) for part in parts)
        if lettered and all(_sides(part) is not None for part in parts):
            return True
    return False


def _is_check(clause: str, problem: _Kind, expected: Answers | None = None) -> bool:
    """Whether a line of arithmetic with no letter is a check or scratch work, not the answer: every such line for a
    problem with a letter ("3(5) + 5 = 20", "15 + 5 = 20", "3(4) + 5 = 17" for 3x + 5 = 20), and for arithmetic one
    that does not start from the problem's value ("3 × 4 = 12" for 12 ÷ 4). "1/2 + 1/3 = 5/6" and "3/6 + 2/6 = 5/6"
    are the problem 1/2 + 1/3 worked out: the answer. For a problem code can't read, a true line ("3(5) + 5 = 20")
    is a check unless it comes to the `expected` value ("3 × 4 = 12" for a rectangle's area of 12); a false one
    ("1/2 + 1/3 = 2/5") is an answer."""
    if problem is None:
        return False
    *steps, last = _LEADING_WORDS.sub("", clause).split("=")
    run, after = _result_run(last)  # what the line comes to, the same value _worked_value reads: "20" of "20 ✓"
    if not steps or not run.strip() or _letter_next(run, after):
        return False  # "3 = x", "6 = 2x": the letter on the right
    sides = [_expression(side.strip()) for side in (*steps, run)]
    if any(side is None for side in sides):
        return False  # a letter in it (a step) or not worked out ("12 ÷ 4 = ?")
    letters, value = problem
    if letters:
        return True
    if value is not None:
        return not _close(sides[0], value)
    return _close(sides[0], sides[-1]) and not any(_close(sides[-1], v) for _, v in expected or ())


def parse_answers(text: Any, problem: Any = None, expected: Answers | None = None) -> Answers:
    """A written or spoken answer as a set of (letter, value): "x = 5 or x = -3" -> {("x", 5), ("x", -3)},
    "x = 5, y = 2" -> {("x", 5), ("y", 2)}, "5" -> {("", 5)}, "(2, 3)" -> {("#1", 2), ("#2", 3)}, "1/2 + 1/3 = 5/6" ->
    {("", 5/6)}. Answers on separate lines all count; a check or a reason after one ("x = 4 because 2 times 4 is 8")
    does not. Steps and problems in the text ("3x + 5 = 20, x = 5") are skipped, and so is arithmetic not worked out
    yet ("1/2 + 1/3", "2/6 + 3/6"); an empty set when nothing reads as an answer.

    Given the `problem` it answers, a line of arithmetic with no letter is a check, never the answer ("x = 5 ;
    3(5) + 5 = 20" is x = 5, "3(5) + 5 = 20" alone is nothing), unless it is that arithmetic problem worked out
    ("1/2 + 1/3 = 5/6" for 1/2 + 1/3). The `expected` answer tells what a problem code can't read asks for: a letter's
    value ("x = 5": every such line is a check) or a value (a true line counts only when it comes to it)."""
    kind = _problem_kind(problem, expected)
    math_text = _answer_text(text)
    if not math_text.strip(" ;"):
        return _NONE
    answers: set[Answer] = set()
    pairs: list[tuple[str, str]] = []

    def keep_pair(found: re.Match[str]) -> str:
        pairs.append((found.group(1), found.group(2)))
        return " ; "

    math_text = _PAIR.sub(keep_pair, math_text)
    for first, second in pairs:
        a, b = _values(first), _values(second)
        if a and b and len(a) == len(b) == 1:
            answers |= {("#1", a[0]), ("#2", b[0])}
    letter = ""
    worked: list[Value] = []
    steps = False
    for raw in _CLAUSES.split(math_text):
        clause = _LABEL.sub("", raw.strip(" .:!?")).strip()
        if not clause:
            continue
        assigned = _assignment(clause) or _right_assignment(clause)
        if assigned is not None:
            letter = assigned[0]
            answers |= {(letter, v) for v in assigned[1]}
            continue
        if "=" in clause:
            if _is_check(clause, kind, expected):
                continue
            value = _worked_value(clause)
            steps = steps or value is None  # the problem or a step ("3x + 5 = 20", "3x = 15")
            worked = value or worked
            continue
        run, after = _value_run(_LEADING_WORDS.sub("", clause))
        if run.strip() and not _letter_next(run, after) and not _unworked(run):
            values = _values(run)
            if values:
                answers |= {(letter, v) for v in values}
    if not answers and worked and not steps:
        # Only arithmetic worked out ("1/2 + 1/3 = 5/6"): its value. Next to a step with a letter, a line like
        # "20 - 5 = 15" is scratch work, not the answer.
        answers = {("", v) for v in worked}
    return frozenset(answers)


def only_checks(text: Any, problem: Any, expected: Answers | None = None) -> bool:
    """Whether `text` holds a check of the answer and no answer: "3(5) + 5 = 20" (or "15 + 5 = 20") for 3x + 5 = 20,
    the bottom line under "x = 5" (`expected` as for parse_answers)."""
    return bool(parse_answers(text)) and not parse_answers(text, problem, expected)


def unfinished(text: Any) -> bool:
    """Whether `text` is arithmetic still to be worked out, not an answer: "1/2 + 1/3", "2/6 + 3/6", "12 ÷ 4 =",
    "one half plus one third". An answer ("5/6", "1/2 + 1/3 = 5/6"), an equation or a step ("3x = 15") is not."""
    if parse_answers(text):
        return False
    for raw in _CLAUSES.split(_answer_text(text)):
        clause = re.sub(r"=\s*[?_\s]*$", "", _LABEL.sub("", raw.strip(" .:!?")).strip()).strip()
        if "=" in clause:
            if _worked_value(clause) == []:
                return True
            continue
        run, after = _value_run(_LEADING_WORDS.sub("", clause))
        if run.strip() and not _letter_next(run, after) and _unworked(run):
            return True
    return False


def _math_key(text: Any) -> str:
    plain = _INSTRUCTION.sub("", _plain(text).strip())
    math_text = re.sub(r"=\s*[?_\s]*$", "", to_math(plain).strip().rstrip(" ?.!"))
    return re.sub(r"\s+", "", math_text)


def same_problem(text: Any, problem: Any) -> bool:
    """Whether `text` is `problem` copied back: the same math, spacing, an instruction ("Simplify") and a trailing
    "= ?" aside ("6/8" for "Simplify 6/8", "1/2+1/3" for "1/2 + 1/3")."""
    key = _math_key(text)
    return bool(key) and key == _math_key(problem)


def _letters(answers: Answers) -> set[str]:
    return {letter for letter, _ in answers if letter}


def _resolve_pairs(answers: Answers, other: Answers) -> Answers:
    """A pair "(2, 3)" against "x = 2, y = 3": its entries are the other side's letters in alphabetical order."""
    named = sorted(letter for letter in _letters(other) if not letter.startswith("#"))
    if not any(letter.startswith("#") for letter, _ in answers) or len(named) != 2:
        return answers
    mapping = {"#1": named[0], "#2": named[1]}
    return frozenset((mapping.get(letter, letter), v) for letter, v in answers)


def _same_values(a: Iterable[Value], b: Iterable[Value]) -> bool:
    left, right = list(a), list(b)
    return all(any(_close(x, y) for y in right) for x in left) and all(any(_close(y, x) for x in left) for y in right)


def compare(student: Answers, expected: Answers) -> bool | None:
    """Whether two parsed answers say the same; None when either is empty or they can't be lined up."""
    if not student or not expected:
        return None
    student, expected = _resolve_pairs(student, expected), _resolve_pairs(expected, student)
    want, got = _letters(expected), _letters(student)
    if len(want) >= 2:  # a system: each letter's value
        if not got:
            return None  # "7, 3": which value is which letter?
        letters = want | got
        return all(
            _same_values([v for x, v in student if x == letter], [v for x, v in expected if x == letter])
            for letter in letters
        )
    if len(got) >= 2:
        return False  # two letters for a one-letter problem
    return _same_values([v for _, v in student], [v for _, v in expected])


def grade(student_text: Any, expected_text: Any) -> bool | None:
    """True / False when the student's answer is / is not the expected one; None when either side can't be read."""
    return compare(parse_answers(student_text), parse_answers(expected_text))


def incomplete(student: Answers, expected: Answers) -> bool:
    """Some of the expected values and none wrong, e.g. "x = 5" for "x = 5 or x = -3"."""
    if not student or not expected or compare(student, expected) is not False:
        return False
    values = [v for _, v in expected]
    return all(any(_close(v, e) for e in values) for _, v in student)


def format_answers(answers: Answers) -> str | None:
    """ "x = 5 or x = -3", "x = 7 and y = 3", "5/6"; None for an empty set."""
    if not answers:
        return None
    items = sorted(answers, key=lambda a: (a[0], -(_number(a[1]) or 0.0)))
    letters = _letters(answers)
    parts = [f"{letter} = {_fmt(v)}" if letter else _fmt(v) for letter, v in items]
    return (" and " if len(letters) >= 2 else " or ").join(parts)


def _fmt(value: Value, *, decimals: bool = False) -> str:
    """ "5", "-3", "5/6" ("2.5" with `decimals`, for a problem written with decimal points), "√13/2"."""
    sp = _sympy()
    if isinstance(value, Fraction) or (sp is not None and isinstance(value, sp.Rational)):
        exact = Fraction(int(value.numerator), int(value.denominator))
        if exact.denominator == 1:
            return str(exact.numerator)
        if decimals and _finite_decimal(exact.denominator):
            return format(float(exact), "g")
        return f"{exact.numerator}/{exact.denominator}"
    return str(value).replace("sqrt", "√").replace("**", "^").replace("*", "")


def _finite_decimal(denominator: int) -> bool:
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    return denominator == 1


# ---- solving the problem ---------------------------------------------------------------------------------------

_INSTRUCTION = re.compile(
    r"^\s*(?:solve|simplify|evaluate|compute|calculate|work\s+out|find|what\s+is|resuelve|calcula|simplifica|"
    r"encuentra|halla|cuanto\s+es)\b(?:\s+(?:for|para)\s+[a-z]\b)?\s*[:,]?\s*"
)
_EQUATIONS = re.compile(r"\s*(?:,(?!\d)|;|\n|&|→|\band\b|(?<=[\w)])\s+y\s+(?=[\w(√-]))\s*")
# Around the math of a worded problem: its sentences ("2x - 7 = 9. Find x."), what comes before the math
# ("despeja x:", "halla x en", "la ecuacion") and words after it ("3x + 5 = 20 para x").
_SENTENCE_BREAK = re.compile(r"[.?!](?:\s+|$)|:")
_WORDS_BEFORE = re.compile(r"^(?:(?:(?!sqrt\b)[a-z]{2,}|[a-z](?=\s+[a-z]{2,}))\s+)+")
_WORDS_AFTER = re.compile(r"\s+(?!sqrt\b)[a-z]{2,}\b.*$")


def _equations(problem: Any) -> list[str] | None:
    # Said in words too ("three x plus five equals twenty"), as a model may copy it off the board.
    text = to_math(_INSTRUCTION.sub("", _plain(problem).strip())).strip().rstrip(" ?.!")
    text = re.sub(r"=\s*[?_\s]*$", "", text)  # "12 ÷ 4 = ?" asks for the value
    if not text or len(text) > MAX_CHARS:
        return None
    if "=" not in text:
        text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", text)  # "6 x 7" is a product, not a letter
    parts = [p for p in _EQUATIONS.split(text) if p.strip()]
    if not parts or len(parts) > MAX_LETTERS:
        return None
    if any(re.search(r"[a-z]{2,}", p.replace("sqrt", "")) for p in parts):
        return None  # words left: a word problem
    return parts


def _sides(part: str) -> tuple[Value, Value] | None:
    sides = part.split("=")
    if len(sides) != 2:
        return None
    left, right = (_expression(s, letters=True) for s in sides)
    return None if left is None or right is None else (left, right)


def solve_text(problem: Any) -> Answers | None:
    """The solution of a board or quiz problem: "3x + 5 = 20" -> {("x", 5)}, "x^2 - 2x - 15 = 0" -> {("x", 5),
    ("x", -3)}, "x + y = 10 and x - y = 4" -> {("x", 7), ("y", 3)}, "1/2 + 1/3" -> {("", 5/6)}. None when there is
    nothing to solve ("x = 5" is solved already) or it is not math code can solve."""
    parts = _equations(problem)
    if parts is None:
        return None
    sp = _sympy()
    if sp is None:
        from biobench_sar.rag.quiz_bank import solve_problem

        solved = solve_problem(" and ".join(parts)) if len(parts) == 1 else None
        return parse_answers(solved) or None
    try:
        return _solve(parts, sp)
    except Exception as exc:  # sympy on odd input: never break a reply over it
        log.debug("could not solve %r: %s", problem, exc)
        return None


def _solve(parts: list[str], sp: Any) -> Answers | None:
    if len(parts) == 1 and "=" not in parts[0]:
        value = _expression(parts[0])
        return frozenset({("", sp.nsimplify(value))}) if value is not None and _number(value) is not None else None
    differences: list[Any] = []
    for part in parts:
        sides = _sides(part)
        if sides is None:
            return None
        differences.append(sides[0] - sides[1])
    if len(parts) == 1:
        left, right = _sides(parts[0]) or (None, None)
        if isinstance(left, sp.Symbol) and _number(right) is not None:
            return None  # "x = 5": already solved
        if isinstance(right, sp.Symbol) and _number(left) is not None:
            return None
    symbols = sorted(set().union(*(d.free_symbols for d in differences)), key=str)
    if not symbols or len(symbols) > MAX_LETTERS or (len(differences) < len(symbols) and len(symbols) > 1):
        return None
    if any(_size(d)[1] > MAX_DEGREE for d in differences):
        return None  # never multiplied out: "(((x + 1)^6)^6)^6 = 0" took seconds (a root counts as a letter here)
    equations = [sp.expand(d) for d in differences]
    degrees: list[int] = []
    rational = True
    for equation in equations:
        try:
            poly = sp.Poly(equation, *symbols)
        except sp.PolynomialError:
            return None
        degree = poly.total_degree()
        degrees.append(degree)
        rational = rational and all(c.is_Rational for c in poly.coeffs())
        unfactored = len(symbols) == 1 and degree > 2 and len(poly.terms()) > 2  # "x^3 = 2" is a power
        if unfactored and any(factor.degree() > 2 for factor, _ in poly.factor_list()[1]):
            return None  # a cubic or quartic that does not factor: its roots in radicals took a third of a second
    # A system with at most one quadratic, the rest linear ("y = x^2 and x + y = 2"), and then whole-number
    # coefficients: sympy took seconds on two quadratics with messy roots, and on "√2 x^2 + √3 y = √5 and √7 x - y = 1".
    # One letter: up to a quartic that factors, or a power ("x^3 = 2").
    curved = sum(degree > 1 for degree in degrees)
    if len(symbols) > 1 and (max(degrees) > 2 or curved > 1 or (curved and not rational)):
        return None
    found: set[Answer] = set()
    for solution in sp.solve(equations, symbols, dict=True):
        if set(solution) != set(symbols):
            continue  # one of a family: not one answer
        values = {str(s): sp.nsimplify(v) for s, v in solution.items()}
        if any(_number(v) is None for v in values.values()):
            continue  # complex
        found |= set(values.items())
    return frozenset(found) or None


def solved_form(text: Any) -> bool:
    """Whether a board line is an answer already ("x = 5", "x = 5 or x = -3"), not a problem to solve."""
    return bool(parse_answers(text)) and solve_text(text) is None


def board_solution(text: Any) -> Answers | None:
    """The solution of the problem a board starts with, whatever else is copied with it: the student's own result
    ("1/2 + 1/3 = 2/5" -> 5/6, worked out from its left side) or steps and an answer below it ("2x + 3 = 11 ; x = 5"
    -> x = 4, "x + y = 10 ; x - y = 4 ; x = 8" -> x = 7 and y = 3). Answers before it are skipped. None when the
    board holds no problem ("x = 5") or none code can solve."""
    found = _board_problem(text)
    return found[1] if found else None


def board_problem(text: Any) -> str | None:
    """The problem a board starts with, as plain math ("1/2 + 1/3" for "1/2 + 1/3 = 2/5", "2x + 3 = 11" for
    "2x + 3 = 11 ; x = 5"); None as for board_solution."""
    found = _board_problem(text)
    return found[0] if found else None


def _board_problem(text: Any) -> tuple[str, Answers] | None:
    parts: list[str] = []
    for raw in _EQUATIONS.split(to_math(_INSTRUCTION.sub("", _plain(text).strip()))):
        clause = raw.strip(" .?!")
        if "=" in clause and not re.search(r"[a-z]", clause.replace("sqrt", "")):
            clause = clause.split("=")[0].strip()  # arithmetic: the problem is the left side, whatever follows
        elif not parts and solved_form(clause):
            continue
        if clause:
            parts.append(clause)
    # One equation, or a system of as many as it has letters: the first that solves is the problem.
    for count in range(1, min(len(parts), MAX_LETTERS) + 1):
        problem = " ; ".join(parts[:count])
        solved = solve_text(problem)
        if solved:
            return problem, solved
    return None


# ---- a reply that states the solution --------------------------------------------------------------------------

_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_ANSWER_WORDS = re.compile(
    r"\b(?:answers?|solutions?|results?|roots?|zeros?|respuestas?|solucion(?:es)?|resultados?|raices|raiz)"
    r"\s*(?:=|:|\bare\b|\bson\b)\s*"
)
_EQUALS = re.compile(r"=\s*")
# A result said without "is": "divide 15 by 3 to get 5", "you should get five", "x turns out to be 5", "x ends up
# being 5", "obtienes cinco", "te sale cinco", "x tiene que ser cinco". Marked "⇒": what follows is a result.
_RESULT_WORDS = {
    "en": re.compile(
        r"\b(?:(?:to|you(?:'ll|\s+will|\s+should|\s+would|\s+can)?|we(?:'ll)?|should|will|would|then)\s+)?"
        r"(?:get|gets|got)\b|\bgives?(?:\s+you)?\b|\bmakes?\b|\bturns?\s+out\s+to\s+be\b|"
        r"\bends?\s+up\s+(?:being|as|at|with)\b|\bcomes?\s+(?:out\s+)?to\b|\bworks?\s+out\s+to\b|\bhas\s+to\s+be\b|"
        r"\bleaves?(?:\s+you)?(?:\s+with)?\b"
    ),
    "es": re.compile(
        r"\b(?:obtienes|obtenemos|obtiene|obtendras|sale|salen|da|dan|queda|quedan|resulta(?:\s+ser)?|"
        r"tiene\s+que\s+ser|tendria\s+que\s+ser|"
        r"(?:debe(?:ria)?s?|tiene\s+que|tendria\s+que|va\s+a)\s+(?:dar|salir|quedar)(?:te|nos)?)\b"
    ),
}
# "That makes x five", "eso hace que x sea cinco": the letter's value.
# "Five sixths is the answer": the value before it.
_IS_THE_ANSWER = re.compile(
    r"=\s*(?:(?:the|your|my|our|final|correct|right|real|la|el|tu|mi)\s+)*"
    r"(?:answers?|solutions?|results?|respuestas?|solucion(?:es)?|resultados?)\b"
)
# What a sentence ends on: "so it is four", "it's actually five-sixths", "eso es cuatro", and "es cuatro" alone.
_ENDS_ON = re.compile(
    r"(?:\b(?:it|that|this|eso|esto)\s*(?:=|'s\b)|^\s*(?:(?:so|then|and|entonces|pues|asi\s+que)\s+)?(?:=|son\b))\s*"
    r"(?:(?:actually|really|just|exactly|still|then|now|realmente|entonces|ahora|solo)\s+)?"
)
_MAKES_LETTER = re.compile(r"\b(?:makes?|hace\s+que)\s+([a-z])\s+(?:sea\s+|valga\s+)?(?=[-\d(√.])")
_LEFT_MATH = re.compile(r"[\d\s.+\-*/÷^()√]*$")


def _results(text: str, language: str | None) -> str:
    text = _MAKES_LETTER.sub(r"\1 = ", text)
    for lang in _languages(language):
        text = _RESULT_WORDS[lang].sub(" ⇒ ", text)
    return text


def _worked_left(before: str) -> bool:
    """Whether the text before an "=" ends in arithmetic on numbers ("15 / 3", "then 20 - 5"), not a term with a
    letter ("3 x", "2(x + 1)", "x / 3")."""
    run = _LEFT_MATH.search(before).group()  # type: ignore[union-attr]  # "*$" always matches
    prefix = before[: len(before) - len(run)]
    return not re.search(r"(?<![a-z])[a-z]$", prefix) and bool(re.search(r"\d\s*[-+*/÷^]\s*[\d(√.]", run))


def _stated(sentence: str, solutions: Answers, language: str | None) -> bool:
    math_text = to_math(sentence, language)
    if language != "en":  # "cinco y menos tres" is two values, not 5y - 3; "tres sextos mas dos sextos son ..." is "="
        math_text = re.sub(r"(?<=\d)\s+y\s+(?=[-\d])", " ; ", math_text)
        math_text = re.sub(r"(?<=[\d)])\s+son\b", " =", math_text)
    marked = _results(math_text, language)
    text = marked.replace("⇒", "=")  # same length: a match in one is at the same place in the other
    by_letter: dict[str, list[Value]] = {}
    for letter, value in solutions:
        by_letter.setdefault(letter, []).append(value)
    every = [v for _, v in solutions]
    # An answer written without its letter ("8", "(7, 3)") is any letter's value: "x = 8" gives it away too.
    named = any(letter and not letter.startswith("#") for letter in by_letter)

    def says(found: re.Match[str], wanted: list[Value], *, last: bool = False) -> bool:
        run, after = _value_run(text[found.end() :])
        if not run.strip() or _letter_next(run, after) or (re.search(r"[-+*/÷^(]\s*$", run) and after[:1].isalpha()):
            return False  # "2 times x": a term
        if last and after.strip(" .,;:!?"):
            return False  # "it is 11 on the right": not what it comes to
        values = _values(run)
        return bool(values) and any(_close(v, w) for v in values for w in wanted)

    for found in _IS_THE_ANSWER.finditer(text):
        before = text[: found.start()]
        run = _LEFT_MATH.search(before).group()  # type: ignore[union-attr]  # "*$" always matches
        values = _values(run) if run.strip() and not re.search(r"(?<![a-z])[a-z]\s*$", before[: -len(run)]) else None
        if values and any(_close(v, w) for v in values for w in every):
            return True
    if any(says(found, every, last=True) for found in _ENDS_ON.finditer(text)):
        return True

    for found in _ASSIGNED.finditer(text):
        letter = found.group(1)
        before = text[: found.start()]
        wanted = by_letter.get(letter) or ([] if named else every)
        if not wanted or _TERM_END.search(before):
            continue
        if says(found, wanted):
            return True
    if any(says(found, every) for found in _ANSWER_WORDS.finditer(text)):
        return True
    if "" in by_letter:  # arithmetic: "... = 5/6" anywhere gives the result away
        return any(says(found, by_letter[""]) for found in _EQUALS.finditer(text))
    # The letter's value without the letter: arithmetic worked out to it ("15 divided by 3 is 5", "quince entre tres
    # es cinco") or a result ("divide 15 by 3 to get 5", "you should get five", "obtienes cinco").
    for found in re.finditer(r"[=⇒]\s*", marked):
        before = marked[: found.start()]
        if found.group().startswith("⇒"):
            if re.search(r"(?<![a-z])[a-z]\s*$", before):
                continue  # "x ⇒ 5" is x's value, read above; "y ⇒ 5" is another letter's
        elif not _worked_left(before):
            continue  # "3 x = 15", "the constant = 5": a step or a word, not a result
        if says(found, every):
            return True
    return False


def solution_sentences(text: Any, solutions: Answers | None, language: str | None = None) -> list[str]:
    """The sentences of a reply that state the problem's solution as an answer ("x = 5", "x is 5", "x equals five",
    "la respuesta es cinco", "x es igual a 5") or as a result ("15 divided by 3 is 5", "you should get five", "x turns
    out to be 5", "obtienes cinco"), as written; [] without solutions."""
    if not solutions:
        return []
    sentences = [s.strip() for s in _SENTENCES.split(str(text or "").strip()) if s.strip()]
    return [s for s in sentences if _stated(s, solutions, language)]


def states_solution(text: Any, solutions: Answers | None, language: str | None = None) -> bool:
    return bool(solution_sentences(text, solutions, language))


def without_solution(text: Any, solutions: Answers | None, language: str | None = None) -> str:
    """The reply without the sentences that state the solution."""
    sentences = [s.strip() for s in _SENTENCES.split(str(text or "").strip()) if s.strip()]
    if not solutions:
        return " ".join(sentences)
    return " ".join(s for s in sentences if not _stated(s, solutions, language))


# ---- one hint for the next step --------------------------------------------------------------------------------

HINTS: dict[str, dict[str, str]] = {
    "en": {
        "substitute": "Put {value} in place of {letter} in the other equation.",
        "eliminate": "Add or subtract the two equations so that one letter cancels out.",
        "square_root": "Get {letter}^2 alone, then take the square root of both sides. There are two answers.",
        "factor_pair": "Factor it: look for two numbers that multiply to {c} and add to {b}.",
        "factor": "Move everything to one side so it equals 0, then try to factor it.",
        "brackets": "Start by multiplying out the brackets.",
        "both_sides": "Get all the {letter} terms on one side first.",
        "like_terms": "Combine the {letter} terms first.",
        "subtract": "Subtract {b} from both sides first.",
        "add": "Add {b} to both sides first.",
        "multiply": "Multiply both sides by {k}.",
        "divide": "Divide both sides by {a}.",
        "flip": "Flip the second fraction and multiply.",
        "times": "Multiply the tops together and the bottoms together.",
        "same_bottom": "The bottoms are the same, so work with the tops.",
        "common": "Find a common denominator first. What number do {d1} and {d2} both go into?",
    },
    "es": {
        "substitute": "Pon {value} en lugar de la {letter} en la otra ecuación.",
        "eliminate": "Suma o resta las dos ecuaciones para que se elimine una letra.",
        "square_root": "Deja sola la {letter}^2 y saca la raíz cuadrada en ambos lados. Hay dos respuestas.",
        "factor_pair": "Factoriza: busca dos números que multiplicados den {c} y sumados den {b}.",
        "factor": "Pasa todo a un lado para que quede igual a 0 y luego intenta factorizar.",
        "brackets": "Empieza por quitar los paréntesis multiplicando.",
        "both_sides": "Primero junta todos los términos con {letter} en un solo lado.",
        "like_terms": "Primero junta los términos con {letter}.",
        "subtract": "Primero resta {b} en ambos lados.",
        "add": "Primero suma {b} en ambos lados.",
        "multiply": "Multiplica ambos lados por {k}.",
        "divide": "Divide ambos lados entre {a}.",
        "flip": "Voltea la segunda fracción y multiplica.",
        "times": "Multiplica los numeradores entre sí y los denominadores entre sí.",
        "same_bottom": "Los denominadores son iguales, así que trabaja con los numeradores.",
        "common": "Primero busca un denominador común. ¿Qué número se puede dividir entre {d1} y entre {d2}?",
    },
}
_DENOMINATORS = re.compile(r"\d+\s*/\s*(\d+)")


def hint_for(problem: Any, language: str = "en") -> str | None:
    """One hint for the first step of `problem`, never its answer ("3x + 5 = 20" -> "Subtract 5 from both sides
    first."); None when code can't tell (a word problem, plain whole numbers)."""
    texts = HINTS["es" if language == "es" else "en"]
    parts = _equations(problem)
    sp = _sympy()
    if parts is None or sp is None:
        return None
    try:
        return _hint(parts, texts, sp)
    except Exception as exc:
        log.debug("no hint for %r: %s", problem, exc)
        return None


def _hint(parts: list[str], texts: dict[str, str], sp: Any) -> str | None:
    if len(parts) == 1 and "=" not in parts[0]:
        return _arithmetic_hint(parts[0], texts)
    sides = [_sides(part) for part in parts]
    if any(s is None for s in sides):
        return None
    pairs = [s for s in sides if s is not None]
    if len(pairs) >= 2:
        for left, right in pairs:
            for lone, other in ((left, right), (right, left)):
                if isinstance(lone, sp.Symbol) and other.free_symbols - {lone}:
                    return texts["substitute"].format(value=_fmt(other), letter=lone)
        return texts["eliminate"]
    left, right = pairs[0]
    symbols = sorted((left - right).free_symbols, key=str)
    if len(symbols) != 1 or _size(left - right)[1] > MAX_DEGREE:
        return None
    letter = symbols[0]
    poly = sp.Poly(sp.expand(left - right), letter)
    if poly.degree() == 2:
        a, b, c = (poly.coeff_monomial(letter**n) for n in (2, 1, 0))
        if b == 0:
            return texts["square_root"].format(letter=letter)
        if a == 1 and all(v.is_integer for v in (b, c)) and right == 0:
            return texts["factor_pair"].format(b=_fmt(b), c=_fmt(c))
        return texts["factor"]
    if poly.degree() != 1:
        return None
    raw = parts[0]
    if "(" in raw:
        return texts["brackets"]
    if letter in left.free_symbols and letter in right.free_symbols:
        return texts["both_sides"].format(letter=letter)
    side, side_text = (left, raw.split("=")[0]) if letter in left.free_symbols else (right, raw.split("=")[1])
    if len(re.findall(rf"(?<![a-z]){letter}(?![a-z])", side_text)) >= 2:
        return texts["like_terms"].format(letter=letter)
    linear = sp.Poly(sp.expand(side), letter)
    a, b = linear.coeff_monomial(letter), linear.coeff_monomial(1)
    if b != 0:
        return texts["subtract" if b > 0 else "add"].format(b=_fmt(abs(b), decimals="." in raw))
    if a == 1:
        return None
    if a.is_Rational and a.p == 1 and a.q > 1:
        return texts["multiply"].format(k=a.q)
    return texts["divide"].format(a=_fmt(a, decimals="." in raw))


def _arithmetic_hint(expression: str, texts: dict[str, str]) -> str | None:
    if "÷" in expression or re.search(r"/\s*\(?\s*\d+\s*/", expression):
        return texts["flip"] if _DENOMINATORS.search(expression) else None
    denominators = [int(d) for d in _DENOMINATORS.findall(expression)]
    if not denominators:
        return None
    if "*" in expression:
        return texts["times"]
    if len(set(denominators)) == 1:
        return texts["same_bottom"]
    return texts["common"].format(d1=denominators[0], d2=denominators[1])
