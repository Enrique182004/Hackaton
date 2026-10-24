"""Spell out math notation inside spoken sentences so a TTS engine reads it naturally.

The text is tokenized, maximal math regions (numbers, single-letter variables, operators, groups) are parsed
with a small precedence parser and rendered as words; everything else is copied verbatim.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["has_math_symbols", "to_speech"]


def _split(words: str) -> tuple[str, ...]:
    return tuple(words.split())


_ONES = _split(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand"))
_MAX_INT = 10**15
_ORD_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_MAX_DEPTH = 24
_MAX_PASSES = 4
# Past this many tokens a region is read flat, left to right: rendering its tree would recurse once per term.
_MAX_REGION_TOKENS = 300

_GREEK = {
    "π": "pi",
    "θ": "theta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "Δ": "delta",
    "δ": "delta",
    "λ": "lambda",
    "μ": "mu",
    "σ": "sigma",
    "φ": "phi",
    "ω": "omega",
    "∞": "infinity",
}
_VULGAR = {
    "½": (1, 2),
    "⅓": (1, 3),
    "⅔": (2, 3),
    "¼": (1, 4),
    "¾": (3, 4),
    "⅕": (1, 5),
    "⅖": (2, 5),
    "⅗": (3, 5),
    "⅘": (4, 5),
    "⅙": (1, 6),
    "⅚": (5, 6),
    "⅐": (1, 7),
    "⅛": (1, 8),
    "⅜": (3, 8),
    "⅝": (5, 8),
    "⅞": (7, 8),
    "⅑": (1, 9),
    "⅒": (1, 10),
}
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸ"
_SUPER_MAP = str.maketrans(_SUPERSCRIPTS, "0123456789+-=()nixy")
_SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_OPWORDS = {
    "+": "plus",
    "-": "minus",
    "±": "plus or minus",
    "∓": "minus or plus",
    "*": "times",
    "×": "times",
    "·": "times",
    "∙": "times",
    "÷": "divided by",
    "/": "over",
    "=": "equals",
    "==": "equals",
    "≠": "is not equal to",
    "!=": "is not equal to",
    "<": "is less than",
    ">": "is greater than",
    "≤": "is less than or equal to",
    "<=": "is less than or equal to",
    "≥": "is greater than or equal to",
    ">=": "is greater than or equal to",
    "≈": "is approximately equal to",
    "∈": "is in",
    "∉": "is not in",
    "→": "gives",
    "->": "gives",
    "=>": "gives",
    "⇒": "gives",
}
_UNARY = {"-": "negative", "+": "positive", "±": "plus or minus", "∓": "minus or plus"}
_REL = frozenset({"=", "==", "≠", "!=", "<", ">", "≤", "<=", "≥", ">=", "≈", "∈", "∉", "→", "->", "=>", "⇒"})
_ADD = frozenset({"+", "-", "±", "∓"})
_MUL = frozenset({"*", "×", "·", "∙", "÷", "/"})
_OPEN = {"(": ")", "[": "]", "{": "}"}
_OPS = sorted(
    [*_OPWORDS, "**", "^", "%", "°", "√", "∛", "|", "(", ")", "[", "]", "{", "}", ","],
    key=len,
    reverse=True,
)
# Leftover operators that could not be parsed into a region: speak them anyway, never pass the symbol through.
_FALLBACK = {
    **_OPWORDS,
    "-": "-",
    "*": "",
    "**": "",
    "^": "",
    "%": "percent",
    "°": "degrees",
    "√": "square root",
    "∛": "cube root",
    "|": "",
}

_FUNCS = {
    "sqrt": "sqrt",
    "cbrt": "cbrt",
    "abs": "abs",
    "sin": "sine",
    "cos": "cosine",
    "tan": "tangent",
    "log": "log",
    "ln": "natural log",
}
_ROOT_WORDS = {"sqrt": "square root", "cbrt": "cube root"}
_ONE_SCALES = frozenset({"one hundred", "one thousand", "one million"})  # "count by 100s" -> "by hundreds"
_FUNC_LETTERS = frozenset("fgh")
_SCALES_TEMP = {"C": "Celsius", "F": "Fahrenheit"}
# Kept whole after a number ("5cm", "3pm"); any other short letter run after a number is a product of variables.
_UNITS = frozenset(_split("cm mm km kg mg lb lbs oz ft yd mi in ml hr hrs min mins sec secs mph am pm gal qt pt pi"))
_RATE_UNITS = frozenset(
    _split(
        "m km cm mm mi mile miles ft feet meter meters g kg mg lb lbs oz l ml gal h hr hrs hour hours min mins "
        "minute minutes s sec secs second seconds day days week weeks month months year years yr dollar dollars "
        "cent cents item items person each unit units page pages"
    )
)
_SLASH_OR = frozenset(_split("and or his her he she him yes no true false either on off in out up down plus minus"))
# Two or three letters next to an operator are read as separate variables ("mx" -> "m x") unless they are a word.
_SHORT_WORDS = frozenset(
    _split(
        "a i am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we add all and any "
        "are ask bad big but can did end few for get got had has her him his how its let lot new not now odd "
        "off old one our out own put say see set she sum the too top try two use was way who why yes yet you "
        "pi"
    )
)
_POINT_WORDS = frozenset({"point", "points", "coordinate", "coordinates", "pair"})
# "Problem 3/10" is a position and "You got 7/10" a score, not fractions; "problems 1-5" is a range.
_OF_WORDS = {"problem": "of", "question": "of", "exercise": "of", "step": "of", "got": "out of", "scored": "out of"}
_RANGE_WORDS = frozenset({"problems", "questions", "exercises", "pages", "steps", "items", "numbers", "ages"})
# "grades 7-8", "Score: 3-2" are ranges or scores whatever the order; "from 1-10" only when it counts up.
_RANGE_BEFORE = _RANGE_WORDS | frozenset(
    _split("grade grades chapter chapters section sections lesson lessons level levels line lines page score scores")
)
_RANGE_UP_BEFORE = frozenset({"from", "between"})
# "2-3 more problems", "15-35 meters": a tight count-up followed by a plural noun is a range.
_RANGE_AFTER = _RANGE_WORDS | frozenset(
    _split(
        "more times minutes hours seconds days weeks months years meters miles feet inches centimeters "
        "kilometers grams kilograms liters cups dollars cents people students kids points degrees percent units"
    )
)
_LABEL_AFTER = frozenset({"triangle", "triangles"})  # "a 3-4-5 triangle"
_ARTICLES = frozenset({"a", "A", "I"})
_QUESTION_AUX = frozenset({"is", "are", "was", "were"})
_DO_AUX = frozenset({"does", "did"})  # not "do": "Do 2x + 3 = 11 next" is an instruction

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_TIME_RE = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:])(?:\s?[AaPp]\.?[Mm]\b\.?)?")
_RATIO_RE = re.compile(r"\d+(?:\.\d+)?(?:(?::| : )\d+(?:\.\d+)?)+(?![\w:]|\.\d)")
_MIXED_RE = re.compile(r"(\d+) (\d+)/(\d+)(?![\w./^(])")
_ORD_RE = re.compile(r"\d+(?:st|nd|rd|th)\b")
_PLURAL_RE = re.compile(r"\d+s\b(?!\s*[-+*/^=<>×÷·(])")
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+")
_MONEY_RE = re.compile(r"\$(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.(\d+))?")
_SUBSCRIPT_RE = re.compile(r"_(\d+|[A-Za-z])\b")
_SUB_RUN_RE = re.compile(r"(?<=[A-Za-z])[₀-₉]+")
_EMPHASIS_RE = re.compile(r"(?<![\w*])\*{1,2}(?=\S)([^*\n]*?\S)\*{1,2}(?![\w*])")

_SUPER_RUN_RE = re.compile(f"[{_SUPERSCRIPTS}]+")
_VULGAR_RE = re.compile(f"(\\d?)([{''.join(_VULGAR)}])(?=(\\d)?)")
_EN_DASH_RE = re.compile(r"(?<=[\d)])(\s*)[–—](\s*)(?=[\d(√])|(?<=\b[A-Za-z])(\s*)[–—](\s*)(?=[\d(])")
_DASH_NUM = r"\d+(?:\.\d+)?"
_RANGE_DASH_RE = re.compile(rf"(?<![\w.,–—-])({_DASH_NUM})[–—]({_DASH_NUM})(?![\w(–—-]|[.,]\d)")
_DASH_OPS = frozenset("+-*/^=<>×÷·±≤≥≠≈")
_ROOT_WORD_RE = re.compile(r"\b(?:sqrt|cbrt)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\d+(?:\.\d+){2,}")
# LaTeX the model sometimes writes even when told to use words: \( \frac{2}{3} \times \frac{3}{5} \)
_LATEX_OPEN_RE = re.compile(r"\\[(\[]\s*|\$\$\s*")
_LATEX_CLOSE_RE = re.compile(r"\s*\\[)\]]|\s*\$\$")
_LATEX_INLINE_RE = re.compile(r"(?<![\w$])\$(?![\s\d$])([A-Za-z]|[^$\n]*?[\\^_={}+<>][^$\n]*?)(?<!\s)\$(?![\w$])")
_LATEX_TEXT_RE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*(?:\[\s*(\d)\s*\])?\s*\{([^{}]*)\}")
_LATEX_SCRIPT_RE = re.compile(r"([\^_])\{([^{}]*)\}")
_LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+|[,;:! %{}])")
_LATEX_SYMBOLS = {
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "pm": "±",
    "mp": "∓",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "neq": "≠",
    "approx": "≈",
    "in": "∈",
    "notin": "∉",
    "to": "→",
    "rightarrow": "→",
    "Rightarrow": "⇒",
    "infty": "∞",
    "pi": "π",
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "circ": "°",
    "cdots": "...",
    "ldots": "...",
    "dots": "...",
    "sqrt": "√",
    "left": "",
    "right": "",
    "quad": " ",
    "qquad": " ",
    ",": " ",
    ";": " ",
    ":": " ",
    " ": " ",
    "!": "",
    "%": "%",
    "{": "{",
    "}": "}",
}
_AUX_RE = re.compile(r"(?:^\s*|([.!?;:,(\n\"“])\s*)(is|are|was|were|does|did)\s+$", re.IGNORECASE)
_XY_PLANE_RE = re.compile(r"\b([xyz])-([xyz])(?=[- ]plane\b)")
_POW_STARS_RE = re.compile(r"(?<=[\w)])\s*\*\*\s*(?=[\w(-])")
_UNIT_PER_RE = re.compile(r"(\d\s?)([A-Za-z]+)/([A-Za-z]+)\b")
_NUM_PER_RE = re.compile(r"(\d)\s?/\s?([A-Za-z]{2,})\b")
_WORD_SLASH_RE = re.compile(r"\b([A-Za-z]{2,})/([A-Za-z]+)\b")

_CANDIDATE_RE = re.compile(r"[\\(\d\-−–—/^*=+<>×÷·∙⋅√∛≤≥≠≈∈∉±∓%°|$#→⇒²³¹⁰⁴-⁹⁺⁻⁽⁾ⁿ½⅓-⅞πθαβγΔδλμσφω∞_₀-₉]")
_SYMBOL_RE = re.compile(
    r"[\^*=+<>×÷·∙⋅√∛≤≥≠≈∈∉±∓%°/−²³¹⁰⁴-⁹ⁿ½⅓-⅞→⇒]"
    r"|\\(?:[dt]?frac|sqrt|times|div|cdot|pm|leq?|geq?|neq?|approx|left|right|text|pi)(?![A-Za-z])|\\[()\[\]]"
    r"|\|[^|\n]+\|"
    r"|(?<![\w-])-\s?(?=[\d.(√])"
    r"|\d\s?-"
    r"|\)\s?-\s?[\d(]"
    r"|\b[A-Za-z]\s?-\s?(?:[\d(]|[A-Za-z]\b)"
    r"|(?<![\w-])-[A-Za-z]\b(?!['’-])"
)


def has_math_symbols(text: str) -> bool:
    """True when ``text`` holds operator notation a TTS engine would misread or the spoken-output rule forbids.

    Operators, relation signs, roots, powers/superscripts, slashes, percent/degree signs, vulgar fractions,
    absolute-value bars and minus signs used as math count; plain digits, prose hyphens ("x-axis",
    "step-by-step") and clock times do not.
    """
    return bool(text) and _SYMBOL_RE.search(text) is not None


def to_speech(text: str) -> str:
    """Return ``text`` with every math expression written out in words; prose is left untouched.

    "3x + 5 = 12" -> "three x plus five equals twelve", "2/3" -> "two thirds", "x^2" -> "x squared".
    Idempotent: words in, the same words out. Never raises.
    """
    if not text or not (_CANDIDATE_RE.search(text) or _ROOT_WORD_RE.search(text)):
        return text
    try:
        # Normal sentences settle in one pass; the extra passes only make malformed leftovers idempotent.
        out = text
        for _ in range(_MAX_PASSES):
            nxt = _convert(out)
            if nxt == out:
                break
            out = nxt
        return out
    except Exception as exc:
        logger.warning("mathspeak failed (%s: %s); falling back to symbol replacement", type(exc).__name__, exc)
        return _crude(text)


# ---------------------------------------------------------------------------------------------------- numbers


def _int_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {_int_words(rest)}" if rest else "")
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return f"{_int_words(head)} {name}" + (f" {_int_words(rest)}" if rest else "")
    raise ValueError(n)


def _digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _number_words(s: str) -> str:
    whole, dot, frac = s.replace(",", "").partition(".")
    if not whole:
        head = ""
    elif (len(whole) > 1 and whole[0] == "0") or len(whole) > 15:
        head = _digits(whole)
    else:
        head = _int_words(int(whole))
    if not dot:
        return head
    return f"{head} point {_digits(frac)}".strip()


def _ordinal(n: int) -> str:
    if n >= _MAX_INT:
        return _digits(str(n)) + "th"
    *rest, last = _int_words(n).split()
    if last in _ORD_IRREGULAR:
        last = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join([*rest, last])


def _plural(words: str) -> str:
    if words.endswith("y"):
        return words[:-1] + "ies"
    if words.endswith(("x", "s")):
        return words + "es"
    return words + "s"


def _decade_words(s: str) -> str | None:
    """ "1990" -> "nineteen ninety", "1900" -> "nineteen hundred": the year reading used for "the 1990s"."""
    if len(s) != 4 or s[0] == "0" or s[-1] != "0":
        return None
    n = int(s)
    head, rest = divmod(n, 100)
    if n % 1000 == 0:
        return _int_words(n)
    if rest == 0:
        return f"{_int_words(head)} hundred"
    return f"{_int_words(head)} {_int_words(rest)}"


def _denominator(d: int, plural: bool) -> str:
    if d == 2:
        return "halves" if plural else "half"
    word = _ordinal(d).removeprefix("one ")  # 7/100 -> "seven hundredths"
    return word + "s" if plural else word


def _ordinal_fraction(num: str, den: str) -> str | None:
    if not (num.isdigit() and den.isdigit()):
        return None
    d = int(den)
    if not (2 <= d <= 20 or d in (100, 1000)) or (len(den) > 1 and den[0] == "0"):
        return None
    return f"{_number_words(num)} {_denominator(d, int(num) != 1)}"


def _money_words(dollars: str, cents: str) -> str:
    if cents and len(cents) != 2:
        return f"{_number_words(dollars + '.' + cents)} dollars"
    d = int(dollars.replace(",", ""))
    c = int(cents) if cents else 0
    parts = []
    if d or not c:
        parts.append(f"{_number_words(dollars)} dollar" + ("" if d == 1 else "s"))
    if c:
        parts.append(f"{_int_words(c)} cent" + ("" if c == 1 else "s"))
    return " and ".join(parts)


# -------------------------------------------------------------------------------------------------- tokenizer


@dataclass(frozen=True)
class _Tok:
    kind: str  # ws num mixed ratio money time section ord plural hash var letter word op other
    text: str
    parts: tuple[str, ...] = ()  # var: (spoken name,); mixed: (whole, num, den); ratio: terms; money: (dollars, cents)


def _latex(text: str) -> str:
    """Turn the common LaTeX math commands into the plain notation the parser reads."""
    text = _LATEX_INLINE_RE.sub(r"\1", text)
    text = _LATEX_CLOSE_RE.sub("", _LATEX_OPEN_RE.sub("", text))
    for _ in range(_MAX_DEPTH):  # innermost braces first, so nested fractions unfold one level per pass
        nxt = _LATEX_TEXT_RE.sub(_latex_text, text)
        nxt = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", nxt)
        nxt = _LATEX_SQRT_RE.sub(lambda m: f"{'∛' if m.group(1) == '3' else '√'}({m.group(2)})", nxt)
        nxt = _LATEX_SCRIPT_RE.sub(lambda m: f"_{m.group(2)}" if m.group(1) == "_" else f"^({m.group(2)})", nxt)
        if nxt == text:
            break
        text = nxt
    text = _LATEX_CMD_RE.sub(lambda m: _LATEX_SYMBOLS.get(m.group(1), m[0]), text)
    return re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)


def _latex_text(m: re.Match[str]) -> str:
    """ "3\\text{ and }y" -> "3 and y": keep the words apart from the math around them."""
    s, start, end = m.string, m.start(), m.end()
    before = " " if start and not s[start - 1].isspace() else ""
    after = " " if end < len(s) and not s[end].isspace() else ""
    return f"{before}{m.group(1).strip()}{after}"


def _prepass(text: str) -> str:
    if any(mark in text for mark in ("\\", "$", "^{", "_{")):
        text = _latex(text)
    text = text.replace("−", "-").replace("⋅", "·").replace("∗", "*")
    text = _RANGE_DASH_RE.sub(_range_dash, text)
    text = _EN_DASH_RE.sub(_en_dash, text)
    text = _EMPHASIS_RE.sub(r"\1", text)
    text = _SUPER_RUN_RE.sub(_superscript, text)
    text = _SUB_RUN_RE.sub(lambda m: "_" + m.group().translate(_SUB_MAP), text)
    text = _VULGAR_RE.sub(_vulgar, text)
    text = _XY_PLANE_RE.sub(r"\1 \2", text)
    text = _POW_STARS_RE.sub("^", text)
    text = _UNIT_PER_RE.sub(_unit_per, text)
    text = _NUM_PER_RE.sub(
        lambda m: f"{m.group(1)} per {m.group(2)}" if m.group(2).lower() in _RATE_UNITS else m[0], text
    )
    return _WORD_SLASH_RE.sub(_word_slash, text)


def _range_dash(m: re.Match[str]) -> str:
    """ "pages 3–5": an unspaced en/em dash between two numbers is a range unless an operator touches it."""
    before = m.string[: m.start()].rstrip()
    after = m.string[m.end() :].lstrip()
    if before[-1:] in _DASH_OPS or after[:1] in _DASH_OPS:
        return m[0]
    joiner = "and" if re.search(r"\bbetween$", before, re.IGNORECASE) else "to"
    return f"{m.group(1)} {joiner} {m.group(2)}"


def _en_dash(m: re.Match[str]) -> str:
    # Only one alternative takes part in a match; its groups may be "" while the other's are None.
    if m.group(1) is not None:
        return f"{m.group(1)}-{m.group(2)}"
    return f"{m.group(3)}-{m.group(4)}"


def _superscript(m: re.Match[str]) -> str:
    t = m.group().translate(_SUPER_MAP)
    return f"^{t}" if re.fullmatch(r"\d+|[A-Za-z]", t) else f"^({t})"


def _vulgar(m: re.Match[str]) -> str:
    num, den = _VULGAR[m.group(2)]
    text = f"{m.group(1)} {num}/{den}" if m.group(1) else f"{num}/{den}"
    return f"{text} " if m.group(3) else text


def _unit_per(m: re.Match[str]) -> str:
    a, b = m.group(2), m.group(3)
    if a.lower() in _RATE_UNITS and b.lower() in _RATE_UNITS:
        return f"{m.group(1)}{a} per {b}"
    return m[0]


def _word_slash(m: re.Match[str]) -> str:
    a, b = m.group(1), m.group(2)
    if b.lower() in _RATE_UNITS:
        return f"{a} per {b}"
    if a.lower() in _SLASH_OR and b.lower() in _SLASH_OR:
        return f"{a} {b}" if "or" in (a.lower(), b.lower()) else f"{a} or {b}"
    return m[0]


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        prev = text[i - 1] if i else ""
        if c.isspace():
            m = _WS_RE.match(text, i)
            assert m is not None
            toks.append(_Tok("ws", m.group()))
            i = m.end()
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit() and not (prev.isalnum() or prev == ".")):
            tok, i = _number_token(text, i, prev)
            toks.append(tok)
            continue
        if c == "$" and (m := _MONEY_RE.match(text, i)):
            toks.append(_Tok("money", m.group(), (m.group(1), m.group(2) or "")))
            i = m.end()
            continue
        if c == "#" and i + 1 < n and text[i + 1].isdigit():
            toks.append(_Tok("hash", c))
            i += 1
            continue
        if c in _GREEK:
            toks.append(_Tok("var", c, (_GREEK[c],)))
            i += 1
            continue
        if c.isalpha():
            m = _WORD_RE.match(text, i)
            assert m is not None
            i = _word_tokens(toks, text, m)
            continue
        op = next((o for o in _OPS if text.startswith(o, i)), None)
        if op is not None:
            toks.append(_Tok("op", op))
            i += len(op)
            continue
        toks.append(_Tok("other", c))
        i += 1
    return toks


def _number_token(text: str, i: int, prev: str) -> tuple[_Tok, int]:
    free = not prev or not (prev.isalnum() or prev in ".:^/")
    if free and (m := _TIME_RE.match(text, i)):
        return _Tok("time", m.group()), m.end()
    if free and (m := _RATIO_RE.match(text, i)):
        return _Tok("ratio", m.group(), tuple(re.split(r"\s?:\s?", m.group()))), m.end()
    if free and (m := _MIXED_RE.match(text, i)):
        _, num, den = m.groups()
        if 0 < int(num) < int(den):
            return _Tok("mixed", m.group(), m.groups()), m.end()
    if m := _SECTION_RE.match(text, i):
        return _Tok("section", m.group()), m.end()
    if m := _ORD_RE.match(text, i):
        return _Tok("ord", m.group()), m.end()
    if m := _PLURAL_RE.match(text, i):
        return _Tok("plural", m.group()), m.end()
    m = _NUM_RE.match(text, i)
    assert m is not None
    return _Tok("num", m.group()), m.end()


def _word_tokens(toks: list[_Tok], text: str, m: re.Match[str]) -> int:
    w, end = m.group(), m.end()
    after_number = bool(toks) and toks[-1].kind == "num"
    if after_number and w.isascii() and w.isalpha() and 1 < len(w) <= 3 and w.lower() not in _UNITS:
        toks.extend(_Tok("letter", ch) for ch in w)
        return end
    if len(w) == 1 and w.isascii():
        sub = _SUBSCRIPT_RE.match(text, end)
        if sub:
            index = sub.group(1)
            spoken = f"{w} sub {_number_words(index) if index.isdigit() else index}"
            toks.append(_Tok("var", w + sub.group(), (spoken,)))
            return sub.end()
        toks.append(_Tok("letter", w))
        return end
    toks.append(_Tok("word", w))
    return end


# ----------------------------------------------------------------------------------------------------- parser


@dataclass(frozen=True)
class _Node:
    kind: str  # num mixed ratio money var word neg bin imp frac pow root abs func pct deg group list
    text: str = ""
    kids: tuple[_Node, ...] = ()
    parts: tuple[str, ...] = ()


class _Fail(Exception):
    pass


_ATOM_KINDS = frozenset({"num", "mixed", "ratio", "money", "var", "letter", "word"})
_PREFIX_OPS = frozenset({"(", "[", "{", "|", "√", "∛"})


class _Parser:
    def __init__(self, toks: list[_Tok], start: int) -> None:
        self.toks = toks
        self.start = start
        self.i = start
        self.abs_depth = 0
        self.depth = 0

    def _skip(self, j: int) -> int:
        while j < len(self.toks) and self.toks[j].kind == "ws":
            j += 1
        return j

    def _peek(self) -> tuple[_Tok | None, int, bool]:
        j = self._skip(self.i)
        return (self.toks[j] if j < len(self.toks) else None), j, j > self.i

    def _adjacent(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _is_op(self, tok: _Tok | None, ops: frozenset[str] | set[str] | dict[str, str]) -> bool:
        return tok is not None and tok.kind == "op" and tok.text in ops

    def _starts_operand(self, tok: _Tok | None) -> bool:
        if tok is None:
            return False
        if tok.kind in _ATOM_KINDS:
            return True
        return tok.kind == "op" and (tok.text in _PREFIX_OPS or tok.text in _UNARY)

    def _try(self, parse_fn: Callable[[], _Node]) -> _Node | None:
        save = (self.i, self.abs_depth, self.depth)
        try:
            return parse_fn()
        except _Fail:
            self.i, self.abs_depth, self.depth = save
            return None

    def relation(self) -> _Node:
        left = self.additive()
        while True:
            tok, j, _ = self._peek()
            if tok is None or not self._is_op(tok, _REL):
                return left
            save, self.i = self.i, j + 1
            right = self._try(self.additive)
            if right is None:
                self.i = save
                return left
            left = _Node("bin", tok.text, (left, right))

    def additive(self) -> _Node:
        left = self.product()
        while True:
            tok, j, ws_before = self._peek()
            if tok is None or not self._is_op(tok, _ADD):
                return left
            if tok.text == "-" and not self._binary_minus(left, j, ws_before):
                return left
            tight = not ws_before and not self._ws_at(j + 1)
            save, self.i = self.i, j + 1
            right = self._try(self.product)
            if right is None:
                self.i = save
                return left
            left = _Node("bin", tok.text, (left, right), parts=("tight",) if tight else ())

    def _binary_minus(self, left: _Node, j: int, ws_before: bool) -> bool:
        k = self._skip(j + 1)
        if k >= len(self.toks):
            return False
        nxt, ws_after = self.toks[k], k > j + 1
        if nxt.kind == "word" and nxt.text.lower() not in _FUNCS:
            return False  # prose dash or hyphenated compound: "x-axis", "3-digit", "work - now"
        if not self._starts_operand(nxt):
            return False
        if left.kind == "var" and nxt.kind == "num" and self.i == self.start + 1 and self._ends_phrase(k):
            if ws_before and not ws_after and left.text in _ARTICLES:
                return False  # "You get a -3": the article and a negative number
            if not (ws_before or ws_after) and left.text.isupper() and nxt.text.isdigit():
                return False  # "Vitamin B-12"
        # After a word only "length - 2" subtracts: "COVID-19" is a label and "is -4" a negative number.
        return left.kind != "word" or (ws_before and ws_after)

    def _ws_at(self, j: int) -> bool:
        return j < len(self.toks) and self.toks[j].kind == "ws"

    def _ends_phrase(self, k: int) -> bool:
        """The number at ``k`` is not followed by more math: "a -3 on the right", not "a -3 = -1"."""
        if k + 1 < len(self.toks) and self.toks[k + 1].kind in ("letter", "var"):
            return False
        nxt = self._skip(k + 1)
        return nxt >= len(self.toks) or self.toks[nxt].kind != "op"

    def product(self) -> _Node:
        left = self._term()
        while True:
            tok, j, ws = self._peek()
            if tok is not None and self._is_op(tok, _MUL):
                spaced = ws or (j + 1 < len(self.toks) and self.toks[j + 1].kind == "ws")
                # "3/4x" is three fourths of x, but "2y / 8y" and "4y ÷ 8x" divide by the whole term.
                tight = tok.text == "/" and not spaced
                save, self.i = self.i, j + 1
                right = self._try(self.unary if tight else self._fraction_term)
                if right is None:
                    self.i = save
                    return left
                if tok.text == "/":
                    node = _Node("frac", "/", (left, right), parts=("spaced",) if spaced else ())
                else:
                    node = _Node("bin", tok.text, (left, right))
                left = self._implicit_chain(node)
                continue
            if ws and tok is not None and tok.text == "x" and tok.kind == "letter" and self._spaced_times(left, j):
                self.i = j + 1
                left = _Node("bin", "×", (left, self._fraction_term()))
                continue
            return left

    def _fraction_term(self) -> _Node:
        """A term plus any tight fraction bars after it: the right side of "2/3 × 3/5" is "3/5", not "3"."""
        left = self._term()
        while self._is_op(self._adjacent(), {"/"}) and not self._ws_at(self.i + 1):
            save, self.i = self.i, self.i + 1
            right = self._try(self.unary)
            if right is None:
                self.i = save
                return left
            left = self._implicit_chain(_Node("frac", "/", (left, right)))
        return left

    def _term(self) -> _Node:
        return self._implicit_chain(self.unary())

    def _implicit_chain(self, left: _Node) -> _Node:
        while True:
            nxt = self._adjacent()
            if nxt is None or not self._implicit_ok(left, nxt):
                return left
            right = self._try(self.power)
            if right is None:
                return left
            left = _Node("imp", "", (left, right))

    def _spaced_times(self, left: _Node, j: int) -> bool:
        if _strip_neg(left).kind not in ("num", "mixed"):
            return False
        k = j + 1
        if k >= len(self.toks) or self.toks[k].kind != "ws":
            return False
        k = self._skip(k)
        return k < len(self.toks) and self.toks[k].kind in ("num", "mixed")

    def _implicit_ok(self, left: _Node, nxt: _Tok) -> bool:
        if _strip_neg(left).kind in ("word", "pct", "money", "ratio", "mixed", "deg"):
            return False
        if nxt.kind in ("letter", "var"):
            return True
        if nxt.kind == "word":
            return nxt.text.lower() in _FUNCS
        if nxt.kind == "op":
            return nxt.text in ("(", "[", "{", "√", "∛") or (nxt.text == "|" and self.abs_depth == 0)
        return False

    def unary(self) -> _Node:
        tok, j, _ = self._peek()
        if tok is None or not self._is_op(tok, _UNARY):
            return self.power()
        k = self._skip(j + 1)
        nxt = self.toks[k] if k < len(self.toks) else None
        if nxt is None or not self._starts_operand(nxt):
            raise _Fail
        if nxt.kind == "word" and nxt.text.lower() not in _FUNCS:
            raise _Fail
        if k > j + 1 and nxt.kind in ("letter", "word"):
            raise _Fail  # "- a list item", not "negative a"
        self.i = j + 1
        self._enter()
        child = self.unary()
        self.depth -= 1
        return _Node("neg", tok.text, (child,))

    def power(self) -> _Node:
        base = self.postfix()
        tok, j, _ = self._peek()
        if not self._is_op(tok, {"^"}):
            return base
        save, self.i = self.i, j + 1
        self._enter()
        exp = self._try(self.unary)
        self.depth -= 1
        if exp is None:
            self.i = save
            return base
        return _Node("pow", "", (base, exp))

    def postfix(self) -> _Node:
        node = self.primary()
        while True:
            tok, j, _ = self._peek()
            if self._is_op(tok, {"%"}):
                node = _Node("pct", "", (node,))
            elif self._is_op(tok, {"°"}):
                self.i = j + 1
                scale = self._adjacent()
                if scale is not None and scale.kind == "letter" and scale.text in _SCALES_TEMP:
                    self.i += 1
                    node = _Node("deg", _SCALES_TEMP[scale.text], (node,))
                else:
                    node = _Node("deg", "", (node,))
                continue
            else:
                return node
            self.i = j + 1

    def primary(self) -> _Node:
        tok, j, _ = self._peek()
        if tok is None:
            raise _Fail
        self.i = j + 1
        if tok.kind == "num":
            return _Node("num", tok.text)
        if tok.kind in ("mixed", "ratio", "money"):
            return _Node(tok.kind, tok.text, parts=tok.parts)
        if tok.kind == "var":
            return _Node("var", tok.parts[0])
        if tok.kind == "letter":
            if tok.text in _FUNC_LETTERS and self._is_op(self._adjacent(), {"("}):
                return _Node("func", tok.text, (self._group(),))
            return _Node("var", tok.text)
        if tok.kind == "word":
            low = tok.text.lower()
            if low in _FUNCS and self._is_op(self._adjacent(), _OPEN):
                arg = self._group()
                if low in ("sqrt", "cbrt"):
                    return _Node("root", "2" if low == "sqrt" else "3", (arg,))
                if low == "abs":
                    return _Node("abs", "", (arg,))
                return _Node("func", _FUNCS[low], (arg,))
            if low in _ROOT_WORDS and self._bare_root_arg():
                self._enter()
                child = self.power()
                self.depth -= 1
                return _Node("root", "2" if low == "sqrt" else "3", (child,))
            return _Node("word", tok.text)
        if tok.kind == "op":
            if tok.text in _OPEN:
                self.i = j
                return self._group()
            if tok.text == "|":
                return self._abs()
            if tok.text in ("√", "∛"):
                self._enter()
                child = self.unary()
                self.depth -= 1
                return _Node("root", "2" if tok.text == "√" else "3", (child,))
        raise _Fail

    def _bare_root_arg(self) -> bool:
        """ "sqrt 9", "sqrt x": a root written without parentheses."""
        nxt = self._peek()[0]
        if nxt is None:
            return False
        return nxt.kind in ("num", "mixed") or (nxt.kind in ("letter", "var") and nxt.text not in _ARTICLES)

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise _Fail

    def _group(self) -> _Node:
        tok, j, _ = self._peek()
        if tok is None or tok.text not in _OPEN:
            raise _Fail
        close = _OPEN[tok.text]
        self.i = j + 1
        self._enter()
        saved_abs, self.abs_depth = self.abs_depth, 0
        items = [self.relation()]
        while self._is_op(self._peek()[0], {","}):
            self.i = self._peek()[1] + 1
            items.append(self.relation())
        end, k, _ = self._peek()
        if not self._is_op(end, {close}):
            raise _Fail
        self.i = k + 1
        self.abs_depth = saved_abs
        self.depth -= 1
        inner = items[0] if len(items) == 1 else _Node("list", "", tuple(items))
        return _Node("group", tok.text, (inner,))

    def _abs(self) -> _Node:
        self._enter()
        self.abs_depth += 1
        inner = self.relation()
        end, k, _ = self._peek()
        if not self._is_op(end, {"|"}):
            raise _Fail
        self.i = k + 1
        self.abs_depth -= 1
        self.depth -= 1
        return _Node("abs", "", (inner,))


# --------------------------------------------------------------------------------------------------- renderer


def _strip_neg(n: _Node) -> _Node:
    while n.kind == "neg":
        n = n.kids[0]
    return n


def _unwrap(n: _Node) -> _Node:
    while n.kind == "group" and n.kids[0].kind != "list":
        n = n.kids[0]
    return n


def _is_compound(n: _Node) -> bool:
    return n.kind in ("bin", "list")


def _is_relation(n: _Node) -> bool:
    return n.kind == "bin" and n.text in _REL


def _is_point(n: _Node) -> bool:
    return n.kind == "list" and len(n.kids) == 2 and all(_is_simple(k) for k in n.kids)


def _is_simple(n: _Node) -> bool:
    return _strip_neg(_unwrap(n)).kind == "mixed" or _plain(n)


def _join(left: tuple[str, bool], word: str, right: tuple[str, bool]) -> tuple[str, bool]:
    lt, open_left = left
    rt, open_right = right
    sep = ", " if open_left else " "
    return f"{lt}{sep}{word} {rt}" if word else f"{lt}{sep}{rt}", open_right


def _operand(n: _Node) -> tuple[str, bool]:
    """Render a child of a root/abs/function/exponent; compound content becomes "the quantity ..."."""
    inner = _unwrap(n)
    if inner.kind == "group" and inner.text == "(":
        inner = inner.kids[0]  # f(x, y): arguments, not a point
    if inner.kind == "list":
        return _list(inner, point=False)
    if _is_compound(inner):
        return f"the quantity {_render(inner)[0]}", True
    return _render(inner)


def _list(n: _Node, point: bool) -> tuple[str, bool]:
    text = ", ".join(_render(k)[0] for k in n.kids)
    return (f"the point {text}" if point and _is_point(n) else text), False


def _render(n: _Node) -> tuple[str, bool]:
    """Return (words, open) where open means the text ends inside "the quantity ..." and needs a comma."""
    k = n.kind
    if k == "num":
        return _number_words(n.text), False
    if k == "var":
        return n.text, False
    if k == "word":
        if n.text.lower() in _ROOT_WORDS:
            return _ROOT_WORDS[n.text.lower()], False
        return (" ".join(n.text) if _is_letter_product(n.text) else n.text), False
    if k == "mixed":
        whole, num, den = n.parts
        frac = _ordinal_fraction(num, den) or f"{_number_words(num)} over {_number_words(den)}"
        return f"{_number_words(whole)} and {frac}", False
    if k == "ratio":
        return " to ".join(_number_words(p) for p in n.parts), False
    if k == "money":
        return _money_words(*n.parts), False
    if k == "neg":
        text, is_open = _render(n.kids[0])
        return f"{_UNARY[n.text]} {text}", is_open
    if k == "bin":
        return _join(_render(n.kids[0]), _OPWORDS[n.text], _render(n.kids[1]))
    if k == "imp":
        return _implicit(n.kids[0], n.kids[1])
    if k == "frac":
        return _fraction(n.kids[0], n.kids[1], spaced=bool(n.parts))
    if k == "pow":
        return _power(n.kids[0], n.kids[1])
    if k == "root":
        text, is_open = _operand(n.kids[0])
        return f"the {'square' if n.text == '2' else 'cube'} root of {text}", is_open
    if k == "abs":
        text, is_open = _operand(n.kids[0])
        return f"the absolute value of {text}", is_open
    if k == "func":
        text, is_open = _operand(n.kids[0])
        return f"{n.text} of {text}", is_open
    if k == "pct":
        return _join(_render(n.kids[0]), "", ("percent", False))
    if k == "deg":
        child = _unwrap(n.kids[0])
        unit = "degree" if child.kind == "num" and child.text == "1" else "degrees"
        unit = f"{unit} {n.text}" if n.text else unit
        return _join(_render(n.kids[0]), "", (unit, False))
    if k == "group":
        inner = n.kids[0]
        if inner.kind == "list":
            if n.text == "{":
                return f"the set {_list(inner, point=False)[0]}", False
            return _list(inner, point=n.text == "(")
        if _is_relation(inner):
            return _render(inner)  # "(3x + 5 = 20)" is an equation in parentheses, not a quantity
        if _is_compound(inner):
            return f"the quantity {_render(inner)[0]}", True
        return _render(inner)
    if k == "list":
        return _list(n, point=False)
    raise ValueError(k)


def _is_letter_product(w: str) -> bool:
    """Spot "mx", "bh", "AB": a short run of letters next to an operator that no one would read as a word."""
    if not (1 < len(w) <= 3 and w.isascii() and w.isalpha()) or w.lower() in _SHORT_WORDS:
        return False
    return w.isupper() or not any(v in w.lower() for v in "aeiou")


def _plain(n: _Node) -> bool:
    n = _strip_neg(_unwrap(n))
    if n.kind in ("num", "var"):
        return True
    if n.kind == "pow":
        return _plain(n.kids[0]) and _strip_neg(_unwrap(n.kids[1])).kind in ("num", "var")
    if n.kind == "imp":
        return _plain(n.kids[0]) and _plain(n.kids[1])
    if n.kind == "frac":
        return all(_strip_neg(c).kind in ("num", "var") for c in n.kids)
    return False


def _varlike(n: _Node) -> bool:
    if n.kind == "var":
        return True
    if n.kind == "pow":
        return n.kids[0].kind == "var"
    return False


def _implicit(left: _Node, right: _Node) -> tuple[str, bool]:
    if _plain(left) and _varlike(right):
        return _join(_render(left), "", _render(right))
    return _join(_render(left), "times", _render(right))


def _is_whole(n: _Node) -> bool:
    return n.kind == "num" and n.text.replace(",", "").isdigit()


def _fraction(num: _Node, den: _Node, spaced: bool = False) -> tuple[str, bool]:
    n0, d0 = _unwrap(num), _unwrap(den)
    if spaced and _is_whole(_strip_neg(n0)) and _is_whole(d0):
        return _join(_render(num), "divided by", _render(den))  # "10 / 2" is a division, not ten halves
    signs: list[str] = []
    while n0.kind == "neg":
        signs.append(_UNARY[n0.text])
        n0 = _unwrap(n0.kids[0])
    if n0.kind == "num" and d0.kind == "num" and (words := _ordinal_fraction(n0.text, d0.text)):
        return " ".join([*signs, words]), False  # -2/3 -> "negative two thirds", not "negative two over three"
    return _join(_render(num), "over", _render(den))


def _power(base: _Node, exp: _Node) -> tuple[str, bool]:
    inner = _unwrap(base)
    if base.kind == "group" and inner.kind not in ("num", "var", "word"):
        rendered = (f"the quantity {_render(inner)[0]}", True)  # (-3)^2, (2x)^2: keep the base together
    else:
        rendered = _render(base)
    e = _unwrap(exp)
    if e.kind == "num" and e.text in ("2", "3"):
        return _join(rendered, "", ("squared" if e.text == "2" else "cubed", False))
    return _join(rendered, "to the power of", _operand(exp))


def _render_root(node: _Node, prev_word: str, next_word: str, aux: str = "") -> str:
    if node.kind == "group" and node.text == "(":
        if _is_point(node.kids[0]) and prev_word in _POINT_WORDS:
            return _list(node.kids[0], point=False)[0]
        if _is_relation(node.kids[0]):
            return f"({_render(node.kids[0])[0]})"
    if node.kind == "frac" and len(node.kids) == 2:
        a, b = (_unwrap(k) for k in node.kids)
        whole_numbers = a.kind == b.kind == "num" and a.text.isdigit() and b.text.isdigit()
        if whole_numbers and prev_word in _OF_WORDS and next_word != "of":
            return f"{_number_words(a.text)} {_OF_WORDS[prev_word]} {_number_words(b.text)}"
    if (spoken := _range(node, prev_word, next_word)) is not None:
        return spoken
    if aux and _is_relation(node):
        return _question(node, aux)
    return _render(node)[0]


def _minus_chain(node: _Node) -> tuple[list[str], bool] | None:
    """ "3-4-5" -> (["3", "4", "5"], tight): a chain of whole numbers joined only by minus signs."""
    nums: list[str] = []
    tight = True
    while node.kind == "bin" and node.text == "-":
        right = node.kids[1]
        if right.kind != "num" or not right.text.isdigit():
            return None
        nums.append(right.text)
        tight = tight and bool(node.parts)
        node = node.kids[0]
    if node.kind != "num" or not node.text.isdigit() or not nums:
        return None
    nums.append(node.text)
    return nums[::-1], tight


def _range(node: _Node, prev_word: str, next_word: str) -> str | None:
    """Prose ranges, scores and labels: "problems 1-5", "from 1-10", "2-3 more", "50-50", "a 3-4-5 triangle"."""
    chain = _minus_chain(node)
    if chain is None:
        return None
    nums, tight = chain
    words = [_number_words(x) for x in nums]
    if len(nums) > 2:
        return "-".join(words) if tight and next_word in _LABEL_AFTER else None
    a, b = int(nums[0]), int(nums[1])
    if prev_word in _RANGE_BEFORE:
        return " to ".join(words)
    if prev_word in _RANGE_UP_BEFORE and a < b:
        return f" {'and' if prev_word == 'between' else 'to'} ".join(words)
    if tight and a == b == 50:
        return "fifty-fifty"
    if tight and a < b and next_word in _RANGE_AFTER:
        return " to ".join(words)
    return None


def _question(node: _Node, aux: str) -> str:
    """ "Is 7 > 3?" -> "Is seven greater than three?": the question already says "is"."""
    ops: list[tuple[str, _Node]] = []
    while _is_relation(node):
        ops.append((node.text, node.kids[1]))
        node = node.kids[0]
    acc = _render(node)
    for op, right in reversed(ops):
        word = _OPWORDS[op]
        if op in ("=", "=="):
            word = "equal" if aux in _DO_AUX else "equal to"
        elif aux in _QUESTION_AUX:
            word = word.removeprefix("is ")
        acc = _join(acc, word, _render(right))
    return acc[0]


def _trivial(node: _Node) -> bool:
    if node.kind in ("var", "word"):
        return True
    return node.kind == "group" and node.kids[0].kind in ("var", "word", "num")


# ------------------------------------------------------------------------------------------------------ driver


class _Out:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.last_generated = False

    def _last_char(self) -> str:
        for part in reversed(self.parts):
            if part:
                return part[-1]
        return ""

    def verbatim(self, s: str) -> None:
        if self.last_generated and s[:1].isalnum() and self._last_char().isalnum():
            self.parts.append(" ")
        self.parts.append(s)
        self.last_generated = False

    def generated(self, s: str) -> None:
        if not s:
            return
        if s[:1].isalnum() and self._last_char().isalnum():
            self.parts.append(" ")
        self.parts.append(s)
        self.last_generated = True

    def last_word(self) -> str:
        m = re.search(r"([A-Za-z]+)[:\s]*$", "".join(self.parts[-4:]))
        return m.group(1).lower() if m else ""

    def clause_aux(self) -> str:
        """The auxiliary that opens a question right before the cursor: "Is ", "So, is ", "Does "."""
        window = 12
        m = _AUX_RE.search("".join(self.parts[-window:]))
        if m is None or (m.group(1) is None and len(self.parts) > window):
            return ""
        return m.group(2).lower()

    def text(self) -> str:
        return "".join(self.parts)


def _can_start(toks: list[_Tok], i: int) -> bool:
    tok = toks[i]
    if tok.kind in _ATOM_KINDS:
        return True
    if tok.kind != "op":
        return False
    if tok.text in _PREFIX_OPS:
        return True
    if tok.text in _UNARY:
        prev = toks[i - 1] if i else None
        return prev is None or prev.kind in ("ws", "op", "other")
    return False


def _emit_token(out: _Out, tok: _Tok) -> None:
    if tok.kind == "ord":
        out.generated(_ordinal(int(tok.text[:-2])))
    elif tok.kind == "plural":
        digits = tok.text[:-1]
        words = _decade_words(digits) or _number_words(digits)
        out.generated(_plural(words.removeprefix("one ") if words in _ONE_SCALES else words))
    elif tok.kind == "section":
        out.generated(" point ".join(_number_words(part) for part in tok.text.split(".")))
    elif tok.kind == "word" and tok.text.lower() in _ROOT_WORDS:
        out.generated(_ROOT_WORDS[tok.text.lower()])
    elif tok.kind == "hash":
        out.generated("number")
    elif tok.kind == "var":
        out.generated(tok.parts[0])
    elif tok.kind == "op" and tok.text in _FALLBACK:
        word = _FALLBACK[tok.text]
        if word == tok.text:
            out.verbatim(word)
        elif word:
            out.generated(word)
        else:
            out.last_generated = True  # dropped symbol: keep "a|b" from fusing into "ab"
    else:
        out.verbatim(tok.text)


def _next_word(toks: list[_Tok], i: int) -> str:
    while i < len(toks) and toks[i].kind == "ws":
        i += 1
    return toks[i].text.lower() if i < len(toks) and toks[i].kind in ("word", "letter") else ""


def _convert(text: str) -> str:
    toks = _tokenize(_prepass(text))
    out = _Out()
    i = 0
    while i < len(toks):
        if _can_start(toks, i):
            parser = _Parser(toks, i)
            try:
                node: _Node | None = parser.relation()
            except _Fail:
                node = None
            if node is not None and not _trivial(node):
                if parser.i - i > _MAX_REGION_TOKENS:
                    out.generated(_flat(toks[i : parser.i]))
                else:
                    out.generated(_render_root(node, out.last_word(), _next_word(toks, parser.i), out.clause_aux()))
                i = parser.i
                continue
        _emit_token(out, toks[i])
        i += 1
    return out.text()


_FLAT_WORDS = {**_FALLBACK, **_OPWORDS, "^": "to the power of", "**": "to the power of"}


def _flat(toks: list[_Tok]) -> str:
    """Read a region token by token. Used for regions too long to render as a tree; never recurses."""
    words: list[str] = []
    for tok in toks:
        if tok.kind == "ws":
            continue
        if tok.kind in ("num", "mixed", "ratio", "money"):
            words.append(_render(_Node(tok.kind, tok.text, parts=tok.parts))[0])
        elif tok.kind == "var":
            words.append(tok.parts[0])
        elif tok.kind == "op":
            if word := _FLAT_WORDS.get(tok.text, ""):
                words.append(word)
        else:
            words.append(tok.text)
    return " ".join(words)


def _crude(text: str) -> str:
    for sym, word in sorted(_FALLBACK.items(), key=lambda kv: -len(kv[0])):
        if sym not in "-()[]{},":
            text = text.replace(sym, f" {word} " if word else " ")
    text = _NUM_RE.sub(lambda m: f" {_number_words(m.group())} ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
