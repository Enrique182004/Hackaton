"""Worked examples made and solved in code, for "what is a fraction?" lessons.

A 3B model invents wrong arithmetic when it makes up its own example ("three to twelve fourths ... sixteen fourths").
So the model only says the one-sentence idea and the example comes from here: easy middle-school numbers, every
result computed with exact fractions. The text keeps its notation (1/4 + 2/4 = 3/4); mathspeak reads it out in words
and the screen shows it as written.
"""

from __future__ import annotations

import random
import re
import unicodedata
from fractions import Fraction
from math import gcd, lcm

KINDS = (
    "concept",
    "add",
    "add_unlike",
    "subtract",
    "multiply",
    "divide",
    "equivalent",
    "simplify",
    "mixed",
    "equation",
)

_FRACTION = r"fraction|fraccion|numerator|denominator|numerador|denominador"
# Most specific first: "adding fractions with different denominators" is add_unlike, not add; "denominador" alone is
# the concept. Arithmetic kinds need a fraction in the topic, so "multiplying decimals" gets no fraction example.
_RULES: tuple[tuple[str, str, bool], ...] = (
    ("equation", r"equation|ecuacion|solve for|despej", False),
    ("mixed", r"mixed|improper|mixto|impropi", False),
    (
        "add_unlike",
        r"common denominator|different denominator|unlike denominator|comun denominador|denominador comun|"
        r"(?:distinto|diferente)s? denominador|denominadores (?:distintos|diferentes)",
        False,
    ),
    ("equivalent", r"equivalen", True),
    ("simplify", r"simplif|reduc|lowest terms|minima expresion", True),
    ("divide", r"divid|division", True),
    ("multipl", r"multipl", True),
    ("subtract", r"subtract|\brest(?:a|ar|ando|amos)\b|minus", True),
    ("add", r"\badd|\bsum|addition|\bsuma", True),
    ("concept", _FRACTION, False),
)


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in folded if not unicodedata.combining(c))


def topic_kind(topic: str | None) -> str | None:
    """Which worked example fits a lesson topic ("sumar fracciones" -> "add"), or None when none does."""
    plain = _plain(topic or "")
    if not plain.strip():
        return None
    if plain.strip() in KINDS:
        return plain.strip()
    has_fraction = bool(re.search(_FRACTION, plain))
    for kind, pattern, needs_fraction in _RULES:
        if re.search(pattern, plain) and (has_fraction or not needs_fraction):
            return "multiply" if kind == "multipl" else kind
    return None


def _f(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _sum_to(raw_top: int, bottom: int) -> str:
    """ "5/10 = 1/2" when the sum simplifies, else just "3/4"."""
    raw, reduced = f"{raw_top}/{bottom}", Fraction(raw_top, bottom)
    return raw if _f(reduced) == raw else f"{raw} = {_f(reduced)}"


def _concept(rng: random.Random, es: bool) -> str:
    top, bottom = rng.choice([(1, 2), (3, 4), (2, 3), (3, 8), (5, 6), (2, 5), (1, 4)])
    if es:
        return (
            f"Por ejemplo, partes una pizza en {bottom} rebanadas iguales y tomas {top}: eso es {top}/{bottom}. "
            f"El número de abajo, {bottom}, dice en cuántas partes iguales se partió el entero, y el de arriba, "
            f"{top}, cuántas tomas."
        )
    return (
        f"For example, cut a pizza into {bottom} equal slices and take {top}: that is {top}/{bottom}. The bottom "
        f"number, {bottom}, says how many equal parts the whole has, and the top number, {top}, says how many you take."
    )


def _add(rng: random.Random, es: bool) -> str:
    n = rng.choice([4, 5, 6, 8, 10])
    a = rng.randint(1, n - 2)
    b = rng.randint(1, n - 1 - a)
    if es:
        return (
            f"Por ejemplo, {a}/{n} + {b}/{n}. Los números de abajo son iguales, así que se queda el {n} y sumamos los "
            f"de arriba: {a} + {b} = {a + b}. Entonces {a}/{n} + {b}/{n} = {_sum_to(a + b, n)}."
        )
    return (
        f"For example, {a}/{n} + {b}/{n}. The bottom numbers are the same, so keep the {n} and add the top numbers: "
        f"{a} + {b} = {a + b}. So {a}/{n} + {b}/{n} = {_sum_to(a + b, n)}."
    )


def _add_unlike(rng: random.Random, es: bool) -> str:
    while True:  # a sum under one whole: easier to picture than 17/12
        b, d = rng.choice([(2, 3), (2, 4), (3, 4), (2, 5), (3, 6), (4, 6), (2, 6), (3, 5)])
        a, c = rng.randint(1, b - 1), rng.randint(1, d - 1)
        if Fraction(a, b) + Fraction(c, d) < 1:
            break
    common = lcm(b, d)
    top_a, top_c = a * common // b, c * common // d
    if es:
        return (
            f"Por ejemplo, {a}/{b} + {c}/{d}. Los números de abajo son distintos, así que buscamos un común "
            f"denominador: {common}, porque el {b} y el {d} caben exactos en él. Cambiamos cada fracción: "
            f"{a}/{b} = {top_a}/{common} y {c}/{d} = {top_c}/{common}. Ahora sumamos los de arriba: "
            f"{top_a}/{common} + {top_c}/{common} = {_sum_to(top_a + top_c, common)}."
        )
    return (
        f"For example, {a}/{b} + {c}/{d}. The bottom numbers are different, so find a common denominator: {common}, "
        f"because both {b} and {d} go into it evenly. Change each fraction: {a}/{b} = {top_a}/{common} and "
        f"{c}/{d} = {top_c}/{common}. Now add the tops: {top_a}/{common} + {top_c}/{common} = "
        f"{_sum_to(top_a + top_c, common)}."
    )


def _subtract(rng: random.Random, es: bool) -> str:
    n = rng.choice([4, 5, 6, 8, 10])
    a = rng.randint(2, n - 1)
    b = rng.randint(1, a - 1)
    if es:
        return (
            f"Por ejemplo, {a}/{n} - {b}/{n}. Los números de abajo son iguales, así que se queda el {n} y restamos los "
            f"de arriba: {a} - {b} = {a - b}. Entonces {a}/{n} - {b}/{n} = {_sum_to(a - b, n)}."
        )
    return (
        f"For example, {a}/{n} - {b}/{n}. The bottom numbers are the same, so keep the {n} and subtract the top "
        f"numbers: {a} - {b} = {a - b}. So {a}/{n} - {b}/{n} = {_sum_to(a - b, n)}."
    )


def _multiply(rng: random.Random, es: bool) -> str:
    b, d = rng.randint(2, 6), rng.randint(2, 6)
    a, c = rng.randint(1, b - 1), rng.randint(1, d - 1)
    if es:
        return (
            f"Por ejemplo, {a}/{b} × {c}/{d}. Multiplicamos los de arriba: {a} × {c} = {a * c}. Multiplicamos los de "
            f"abajo: {b} × {d} = {b * d}. Entonces {a}/{b} × {c}/{d} = {_sum_to(a * c, b * d)}."
        )
    return (
        f"For example, {a}/{b} × {c}/{d}. Multiply the tops: {a} × {c} = {a * c}. Multiply the bottoms: "
        f"{b} × {d} = {b * d}. So {a}/{b} × {c}/{d} = {_sum_to(a * c, b * d)}."
    )


def _divide(rng: random.Random, es: bool) -> str:
    while True:  # not "1/2 ÷ 1/2"
        b, d = rng.randint(2, 5), rng.randint(2, 5)
        a, c = rng.randint(1, b - 1), rng.randint(1, d - 1)
        if Fraction(a, b) != Fraction(c, d):
            break
    if es:
        return (
            f"Por ejemplo, {a}/{b} ÷ {c}/{d}. Para dividir entre una fracción, la volteamos y multiplicamos: {c}/{d} "
            f"se vuelve {d}/{c}. Entonces {a}/{b} ÷ {c}/{d} = {a}/{b} × {d}/{c} = {_sum_to(a * d, b * c)}."
        )
    return (
        f"For example, {a}/{b} ÷ {c}/{d}. To divide by a fraction, flip it and multiply: {c}/{d} becomes {d}/{c}. "
        f"So {a}/{b} ÷ {c}/{d} = {a}/{b} × {d}/{c} = {_sum_to(a * d, b * c)}."
    )


def _reduced_pair(rng: random.Random) -> tuple[int, int]:
    while True:
        b = rng.randint(2, 6)
        a = rng.randint(1, b - 1)
        if gcd(a, b) == 1:
            return a, b


def _equivalent(rng: random.Random, es: bool) -> str:
    a, b = _reduced_pair(rng)
    k = rng.randint(2, 4)
    if es:
        return (
            f"Por ejemplo, empezamos con {a}/{b}. Multiplicamos arriba y abajo por el mismo número, {k}: "
            f"{a} × {k} = {a * k} y {b} × {k} = {b * k}. Entonces {a}/{b} = {a * k}/{b * k}: la misma cantidad, "
            f"en más pedazos."
        )
    return (
        f"For example, start with {a}/{b}. Multiply the top and the bottom by the same number, {k}: "
        f"{a} × {k} = {a * k} and {b} × {k} = {b * k}. So {a}/{b} = {a * k}/{b * k}: the same amount, cut into more "
        f"pieces."
    )


def _simplify(rng: random.Random, es: bool) -> str:
    a, b = _reduced_pair(rng)
    k = rng.randint(2, 4)
    top, bottom = a * k, b * k
    if es:
        return (
            f"Por ejemplo, {top}/{bottom}. Buscamos un número que divida arriba y abajo: el {k}. "
            f"{top} ÷ {k} = {a} y {bottom} ÷ {k} = {b}. Entonces {top}/{bottom} = {a}/{b}."
        )
    return (
        f"For example, {top}/{bottom}. Find a number that divides both the top and the bottom: {k}. "
        f"{top} ÷ {k} = {a} and {bottom} ÷ {k} = {b}. So {top}/{bottom} = {a}/{b}."
    )


def _mixed(rng: random.Random, es: bool) -> str:
    d = rng.randint(2, 5)
    q, r = rng.randint(1, 3), rng.randint(1, d - 1)
    n = q * d + r
    if es:
        return (
            f"Por ejemplo, {n}/{d}. Vemos cuántos grupos completos de {d} caben en {n}: {q} × {d} = {q * d}, y sobran "
            f"{r}. Entonces {n}/{d} = {q} y {r}/{d}."
        )
    return (
        f"For example, {n}/{d}. See how many whole groups of {d} fit in {n}: {q} × {d} = {q * d}, with {r} left over. "
        f"So {n}/{d} = {q} and {r}/{d}."
    )


def _equation(rng: random.Random, es: bool) -> str:
    a, x, b = rng.randint(2, 5), rng.randint(1, 9), rng.randint(1, 9)
    c = a * x + b
    if es:
        return (
            f"Por ejemplo, {a}x + {b} = {c}. Primero restamos {b} en ambos lados: {a}x = {c - b}. Luego dividimos "
            f"ambos lados entre {a}: x = {x}. Comprobamos: {a} × {x} + {b} = {c}."
        )
    return (
        f"For example, {a}x + {b} = {c}. First take away {b} from both sides: {a}x = {c - b}. Then divide both sides "
        f"by {a}: x = {x}. Check: {a} × {x} + {b} = {c}."
    )


_MAKERS = {
    "concept": _concept,
    "add": _add,
    "add_unlike": _add_unlike,
    "subtract": _subtract,
    "multiply": _multiply,
    "divide": _divide,
    "equivalent": _equivalent,
    "simplify": _simplify,
    "mixed": _mixed,
    "equation": _equation,
}


def worked_example(topic: str | None, language: str = "en", rng: random.Random | None = None) -> str | None:
    """One easy example on the topic, solved step by step in the reply language, or None for other topics."""
    kind = topic_kind(topic)
    if kind is None:
        return None
    return _MAKERS[kind](rng or random.Random(), str(language or "").lower().startswith("es"))
