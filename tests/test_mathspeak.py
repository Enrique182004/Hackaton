from __future__ import annotations

import random
import re

import pytest

from biobench_sar.llm import mathspeak
from biobench_sar.llm.mathspeak import has_math_symbols, to_speech

CASES: list[tuple[str, str]] = [
    # integers and thousands separators
    ("7", "seven"),
    ("1250", "one thousand two hundred fifty"),
    ("1,250", "one thousand two hundred fifty"),
    ("1,000,000", "one million"),
    ("2026", "two thousand twenty six"),
    ("115", "one hundred fifteen"),
    ("007", "zero zero seven"),
    # negatives: unary vs subtraction
    ("-4", "negative four"),
    ("x - 4", "x minus four"),
    ("x-4", "x minus four"),
    ("x = -4", "x equals negative four"),
    ("3 - -4", "three minus negative four"),
    ("4-x", "four minus x"),
    ("-x", "negative x"),
    ("between -5 and 5", "between negative five and five"),
    ("The answer is -4.", "The answer is negative four."),
    # decimals
    ("0.5", "zero point five"),
    ("3.14", "three point one four"),
    (".5", "point five"),
    ("-0.25", "negative zero point two five"),
    # fractions
    ("1/2", "one half"),
    ("2/3", "two thirds"),
    ("3/4", "three fourths"),
    ("1/3", "one third"),
    ("2/5", "two fifths"),
    ("5/8", "five eighths"),
    ("7/12", "seven twelfths"),
    ("3/20", "three twentieths"),
    ("7/100", "seven hundredths"),
    ("4/2", "four halves"),
    ("1/21", "one over twenty one"),
    ("-2/3", "negative two thirds"),
    ("x/2", "x over two"),
    ("2/x", "two over x"),
    ("(x+1)/3", "the quantity x plus one, over three"),
    ("x/(x+1)", "x over the quantity x plus one"),
    ("(x - 2)/(x + 2)", "the quantity x minus two, over the quantity x plus two"),
    ("3/4x = 6", "three fourths x equals six"),
    ("½", "one half"),
    ("rise/run", "rise over run"),
    # mixed numbers
    ("2 1/2", "two and one half"),
    ("3 3/4", "three and three fourths"),
    ("2½", "two and one half"),
    ("-2 1/2", "negative two and one half"),
    # exponents
    ("x^2", "x squared"),
    ("x^3", "x cubed"),
    ("x^n", "x to the power of n"),
    ("2^10", "two to the power of ten"),
    ("x^-1", "x to the power of negative one"),
    ("x²", "x squared"),
    ("x³", "x cubed"),
    ("x⁴", "x to the power of four"),
    ("x⁻¹", "x to the power of negative one"),
    ("(x+1)²", "the quantity x plus one, squared"),
    ("(-3)^2", "the quantity negative three, squared"),
    ("-x^2", "negative x squared"),
    ("x^(n+1) + 2", "x to the power of the quantity n plus one, plus two"),
    ("2**3 = 8", "two cubed equals eight"),
    # implicit multiplication
    ("3x", "three x"),
    ("3x+5=12", "three x plus five equals twelve"),
    ("2(x+1)", "two times the quantity x plus one"),
    ("2(x + 1) = 10", "two times the quantity x plus one, equals ten"),
    ("(x + 3)(x − 2)", "the quantity x plus three, times the quantity x minus two"),
    ("x(x+1)", "x times the quantity x plus one"),
    ("3(-4)", "three times negative four"),
    ("2x^2 + 3x - 5 = 0", "two x squared plus three x minus five equals zero"),
    ("y = mx + b", "y equals m x plus b"),
    ("y = (1/2)x - 3", "y equals one half x minus three"),
    ("A = πr²", "A equals pi r squared"),
    ("2πr", "two pi r"),
    # operators
    ("3 * 4", "three times four"),
    ("3 × 4", "three times four"),
    ("3·4", "three times four"),
    ("3 x 4 = 12", "three times four equals twelve"),
    ("12 ÷ 4 = 3", "twelve divided by four equals three"),
    ("2 + 3 * 4 = 14", "two plus three times four equals fourteen"),
    ("x = ±3", "x equals plus or minus three"),
    # relations
    ("x ≠ 3", "x is not equal to three"),
    ("x != 3", "x is not equal to three"),
    ("x < 5", "x is less than five"),
    ("x > 5", "x is greater than five"),
    ("x ≤ 5", "x is less than or equal to five"),
    ("x <= 5", "x is less than or equal to five"),
    ("x ≥ -2", "x is greater than or equal to negative two"),
    ("x >= 5", "x is greater than or equal to five"),
    ("-3 < x ≤ 7", "negative three is less than x is less than or equal to seven"),
    ("x ≈ 3.14", "x is approximately equal to three point one four"),
    # percent, degrees, money
    ("25%", "twenty five percent"),
    ("12.5%", "twelve point five percent"),
    ("What's 15% of 80?", "What's fifteen percent of eighty?"),
    ("90°", "ninety degrees"),
    ("It was 72°F outside.", "It was seventy two degrees Fahrenheit outside."),
    ("$5", "five dollars"),
    ("$1.50", "one dollar and fifty cents"),
    ("$0.75", "seventy five cents"),
    # roots and absolute value
    ("sqrt(9)", "the square root of nine"),
    ("√9", "the square root of nine"),
    ("√16 = 4", "the square root of sixteen equals four"),
    ("√(x+1)", "the square root of the quantity x plus one"),
    ("3√2", "three times the square root of two"),
    ("∛8", "the cube root of eight"),
    ("|x|", "the absolute value of x"),
    ("|-5| = 5", "the absolute value of negative five equals five"),
    ("|x − 3| = 5", "the absolute value of the quantity x minus three, equals five"),
    ("|x| + |y|", "the absolute value of x plus the absolute value of y"),
    ("2|x|", "two times the absolute value of x"),
    # functions
    ("f(x) = 2x + 1", "f of x equals two x plus one"),
    ("g(2) = 5", "g of two equals five"),
    ("f(x+1) = 3", "f of the quantity x plus one, equals three"),
    # coordinates
    ("(2, 3)", "the point two, three"),
    ("(-1, 4)", "the point negative one, four"),
    ("(x, y)", "the point x, y"),
    ("Plot the point (2, 3).", "Plot the point two, three."),
    ("It passes through (0, -3).", "It passes through the point zero, negative three."),
    ("(1, 2, 3)", "one, two, three"),
    # ratios vs clock times
    ("3:4", "three to four"),
    ("3:4:5", "three to four to five"),
    ("1:100", "one to one hundred"),
    ("Meet at 10:30.", "Meet at 10:30."),
    ("It's 3:15 pm now.", "It's 3:15 pm now."),
    ("Step 1: subtract 5.", "Step one: subtract five."),
    # ordinals, plurals, labels
    ("the 3rd step", "the third step"),
    ("the 21st problem", "the twenty first problem"),
    ("count by 5s", "count by fives"),
    ("Problem #3", "Problem number three"),
    ("You got 7/10!", "You got seven out of ten!"),
    ("Problem 3/10", "Problem three of ten"),
    ("x₁ + x_2", "x sub one plus x sub two"),
    # prose around expressions stays as written
    ("Solve 3x + 5 = 20 for x.", "Solve three x plus five equals twenty for x."),
    ("If x = 3, what is 2x - 1?", "If x equals three, what is two x minus one?"),
    ("Good work - now subtract 5.", "Good work - now subtract five."),
    ("The x-axis and the y-intercept", "The x-axis and the y-intercept"),
    ("a 3-digit number", "a three-digit number"),
    ("the x-y plane", "the x y plane"),
    ("(5 points)", "(five points)"),
    ("5cm", "five cm"),
    ("60 km/h", "sixty km per h"),
    ("$15/hour", "fifteen dollars per hour"),
    ("and/or", "and or"),
    ("*Great* job!", "Great job!"),
    ("−5°C", "negative five degrees Celsius"),
]

PROSE = [
    "Great job! You solved it.",
    "Laguna de Bustillos is in Chihuahua (Mexico).",
    "Let's go step-by-step, it's a well-known trick.",
    "You're doing great — keep going.",
    "Remember: a negative times a negative is a positive.",
    "Do you have an exercise in mind, or should I make one up for you?",
    "Write the x-axis label (see above) first.",
    "e.g. this, i.e. that",
    "",
]

CORPUS = [
    "Let's solve 3x + 5 = 20 together.",
    "First, subtract 5 from both sides: 3x = 15.",
    "Now divide both sides by 3.",
    "What is x if 2x - 4 = 10?",
    "Look at the fraction 2/3 on the board.",
    "Add 1/2 and 1/4 by finding a common denominator.",
    "Try this one: 5x - 7 = 3x + 9.",
    "The square of x is x^2, so x² + 2x + 1 = (x + 1)².",
    "Remember that -3 * -4 = 12.",
    "Is 7 > -2? Yes!",
    "Solve for y: y/4 = 6.",
    "Simplify (2x + 6)/2.",
    "Problem 2 of 10: 4(x - 1) = 12.",
    "Great! Next problem: x/3 + 2 = 5.",
    "The slope is rise/run = 4/2 = 2.",
    "25% of 80 is 20.",
    "Is √49 equal to 7?",
    "|x - 3| = 5 has two answers.",
    "Plot the point (3, -2) on the grid.",
    "The ratio 2:3 means 2 parts to 3 parts.",
    "If 2 1/2 cups make 5 servings, how much makes 10?",
    "Multiply 0.5 × 12.",
    "Check: 3(4) + 5 = 17. ✓",
    "x ≥ 4 means x can be 4 or bigger.",
    "Expand (x + 2)(x - 2) = x^2 - 4.",
    "Which is bigger, 3/8 or 5/12?",
    "The area is A = πr² ≈ 3.14 × 5².",
    "Write 12 ÷ 4 as a fraction: 12/4.",
    "Find f(2) when f(x) = 3x^2 - x + 1.",
    "Now try 10^3 ÷ 10^2 = 10.",
    "y = -2x + 7 crosses the y-axis at 7.",
    "You scored 8/10 — nice!",
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_to_speech(text: str, expected: str) -> None:
    assert to_speech(text) == expected


def test_case_table_is_large() -> None:
    assert len(CASES) >= 60
    assert len(CORPUS) >= 30


@pytest.mark.parametrize("text", PROSE)
def test_prose_unchanged(text: str) -> None:
    assert to_speech(text) == text


@pytest.mark.parametrize("text", [t for t, _ in CASES] + CORPUS)
def test_idempotent(text: str) -> None:
    once = to_speech(text)
    assert to_speech(once) == once


@pytest.mark.parametrize("text", CORPUS)
def test_corpus_has_no_forbidden_symbols(text: str) -> None:
    spoken = to_speech(text)
    assert not has_math_symbols(spoken), spoken
    assert not re.search(r"[\^/*=+<>%√×÷]|\d", spoken), spoken


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_cases_have_no_forbidden_symbols(text: str, expected: str) -> None:
    assert not re.search(r"[\^/*=+<>%√×÷]", to_speech(text))


@pytest.mark.parametrize(
    "text",
    ["3x + 5 = 12", "x^2", "2/3", "-4", "x - 4", "5 × 3", "√9", "25%", "x ≤ 5", "|x|", "x²", "½", "(x−1)", "a ± b"],
)
def test_has_math_symbols_true(text: str) -> None:
    assert has_math_symbols(text)


@pytest.mark.parametrize(
    "text",
    ["", "Great job!", "the x-axis", "step-by-step", "Meet at 10:30.", "Problem 3", "three x plus five", "COVID"],
)
def test_has_math_symbols_false(text: str) -> None:
    assert not has_math_symbols(text)


def test_has_math_symbols_false_after_speech() -> None:
    for text, _ in CASES:
        assert not has_math_symbols(to_speech(text)), text


def test_none_like_inputs() -> None:
    assert to_speech("") == ""
    assert to_speech("   ") == "   "


def test_whitespace_and_newlines_preserved() -> None:
    assert to_speech("Step 1:\n  x = 2\n\nDone.") == "Step one:\n  x equals two\n\nDone."


def test_leftover_operators_are_spoken() -> None:
    assert to_speech("x =") == "x equals"
    assert to_speech("= 5") == "equals five"
    assert "/" not in to_speech("5 /")
    assert "+" not in to_speech("+")


def test_deep_nesting_does_not_raise() -> None:
    text = "(" * 400 + "x" + ")" * 400 + " = 2"
    out = to_speech(text)
    assert "=" not in out and "2" not in out


def test_unbalanced_groups() -> None:
    assert to_speech("(x + 1 = 3") == "(x plus one equals three"
    assert to_speech("|x + 1 = 3") == "x plus one equals three"  # a stray bar is dropped, never spoken


def test_internal_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: str) -> str:
        raise RuntimeError("bug")

    monkeypatch.setattr(mathspeak, "_convert", boom)
    out = to_speech("x^2 + 3 = 7")
    assert not re.search(r"[\^+=]", out)
    assert "plus" in out and "equals" in out


def test_random_input_never_raises() -> None:
    rng = random.Random(7)
    alphabet = "0123456789xyab+-*/^=()|√%:,. ²½−≤≥$#°"
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        out = to_speech(text)
        assert isinstance(out, str)
        assert to_speech(out) == out, (text, out)
