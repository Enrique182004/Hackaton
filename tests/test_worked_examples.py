import random
import re
from fractions import Fraction

import pytest

from biobench_sar.worked_examples import KINDS, topic_kind, worked_example


@pytest.mark.parametrize(
    ("topic", "kind"),
    [
        ("a fraction", "concept"),
        ("fractions", "concept"),
        ("numerator", "concept"),
        ("una fracción", "concept"),
        ("denominador", "concept"),
        ("adding fractions", "add"),
        ("sumar fracciones", "add"),
        ("a common denominator", "add_unlike"),
        ("adding fractions with different denominators", "add_unlike"),
        ("común denominador", "add_unlike"),
        ("sumar fracciones con distinto denominador", "add_unlike"),
        ("subtracting fractions", "subtract"),
        ("restar fracciones", "subtract"),
        ("multiplying fractions", "multiply"),
        ("multiplicar fracciones", "multiply"),
        ("dividing fractions", "divide"),
        ("dividir fracciones", "divide"),
        ("equivalent fractions", "equivalent"),
        ("las fracciones equivalentes", "equivalent"),
        ("simplifying fractions", "simplify"),
        ("simplificar fracciones", "simplify"),
        ("mixed numbers", "mixed"),
        ("improper fractions", "mixed"),
        ("números mixtos", "mixed"),
        ("an equation", "equation"),
        ("solving equations", "equation"),
        ("resolver ecuaciones", "equation"),
        ("una ecuación", "equation"),
    ],
)
def test_topic_kind(topic: str, kind: str) -> None:
    assert topic_kind(topic) == kind


@pytest.mark.parametrize(
    "topic", ["the area of a triangle", "slope", "probability", "el área de un triángulo", "", None]
)
def test_topics_without_a_worked_example(topic: str | None) -> None:
    # The model only says the idea; no example beats a wrong one.
    assert topic_kind(topic) is None
    assert worked_example(topic) is None


TERM = r"\d+ (?:and|y) \d+/\d+|\d+/\d+|\d+"
SIDE = rf"(?:{TERM})(?:\s*[+\-×÷]\s*(?:{TERM}))*"
CHAIN = re.compile(rf"(?<![\w/]){SIDE}(?:\s*=\s*{SIDE})+(?![\w/])")


def _term(token: str) -> Fraction:
    mixed = re.fullmatch(r"(\d+) (?:and|y) (\d+)/(\d+)", token)
    if mixed:
        return int(mixed[1]) + Fraction(int(mixed[2]), int(mixed[3]))
    top, _, bottom = token.partition("/")
    return Fraction(int(top), int(bottom or 1))


def _value(side: str) -> Fraction:
    """× and ÷ before + and −, left to right: enough for the examples' one-line arithmetic."""
    tokens = re.findall(rf"{TERM}|[+\-×÷]", side)
    sums: list[Fraction] = [_term(tokens[0])]
    signs: list[str] = []
    for op, term in zip(tokens[1::2], tokens[2::2], strict=True):
        value = _term(term)
        if op == "×":
            sums[-1] *= value
        elif op == "÷":
            sums[-1] /= value
        else:
            signs.append(op)
            sums.append(value)
    total = sums[0]
    for sign, value in zip(signs, sums[1:], strict=True):
        total = total + value if sign == "+" else total - value
    return total


def _chains(text: str) -> list[list[Fraction]]:
    return [[_value(side) for side in found.group(0).split("=")] for found in CHAIN.finditer(text)]


ARITHMETIC = [k for k in KINDS if k not in ("concept", "equation")]


@pytest.mark.parametrize("language", ["en", "es"])
@pytest.mark.parametrize("kind", ARITHMETIC)
def test_every_equality_in_an_example_is_true(kind: str, language: str) -> None:
    for seed in range(40):
        text = worked_example(kind, language, random.Random(seed))
        assert text, (kind, language, seed)
        chains = _chains(text)
        assert chains, text
        for chain in chains:
            assert len(set(chain)) == 1, (text, chain)


@pytest.mark.parametrize("language", ["en", "es"])
def test_equation_example_is_solved_and_checked(language: str) -> None:
    for seed in range(40):
        text = worked_example("equations", language, random.Random(seed))
        assert text
        a, b, c = map(int, re.search(r"(\d+)x \+ (\d+) = (\d+)", text).groups())
        x = Fraction(re.findall(r"\bx = (\d+)", text)[-1])
        assert a * x + b == c, text
        assert f"{c - b}" in text  # the middle step: ax = c - b


@pytest.mark.parametrize("language", ["en", "es"])
def test_concept_example_names_the_parts(language: str) -> None:
    for seed in range(20):
        text = worked_example("fractions", language, random.Random(seed))
        top, bottom = map(int, re.search(r"(\d+)/(\d+)", text).groups())
        assert 0 < top < bottom, text
        assert str(bottom) in text.split(f"{top}/{bottom}", 1)[1], text  # the bottom number is explained


def test_examples_are_in_the_reply_language() -> None:
    rng = random.Random(3)
    assert worked_example("adding fractions", "en", rng).startswith("For example")
    assert worked_example("sumar fracciones", "es", rng).startswith("Por ejemplo")


def test_examples_vary_between_asks() -> None:
    examples = {worked_example("adding fractions", "en", random.Random(seed)) for seed in range(10)}
    assert len(examples) >= 3


def test_common_denominator_sums_stay_under_one_and_divisions_differ() -> None:
    for seed in range(60):
        added = _chains(worked_example("add_unlike", "en", random.Random(seed)))[-1][-1]
        assert added < 1, seed
        a, b, c, d = map(
            int, re.search(r"(\d+)/(\d+) ÷ (\d+)/(\d+)", worked_example("divide", "en", random.Random(seed))).groups()
        )
        assert Fraction(a, b) != Fraction(c, d), seed


def test_examples_stay_middle_school_easy() -> None:
    for kind in ARITHMETIC:
        for seed in range(40):
            numbers = [int(n) for n in re.findall(r"\d+", worked_example(kind, "en", random.Random(seed)))]
            assert max(numbers) <= 60, (kind, seed, numbers)
