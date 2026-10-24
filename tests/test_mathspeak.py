from __future__ import annotations

import csv
import logging
import random
import re
from pathlib import Path

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
    ("Maria got 3/4 of the pizza.", "Maria got three fourths of the pizza."),
    ("Try problems 1-5 first.", "Try problems one to five first."),
    ("8-3", "eight minus three"),
    ("3 – 5", "three minus five"),
    ("4y ÷ 8x", "four y divided by eight x"),
    ("2y / 8y", "two y over eight y"),
    ("Plot (x, 5y).", "Plot the point x, five y."),
    ("AB = 5", "A B equals five"),
    ("1/2bh", "one half b h"),
    ("count by 2s", "count by twos"),
    ("2s + 3 = 7", "two s plus three equals seven"),
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

# Review findings: each case failed before the fix it guards.
REVIEW_CASES: list[tuple[str, str]] = [
    # en/em dashes between numbers: no literal "None", unspaced number-dash-number is a range
    ("pages 3–5", "pages three to five"),
    ("Try 2–3 more problems.", "Try two to three more problems."),
    ("5—6", "five to six"),
    ("3 –5", "three minus five"),
    ("(x+1)–2", "the quantity x plus one, minus two"),
    ("x–3", "x minus three"),
    ("between 5–10 minutes", "between five and ten minutes"),
    ("Sections 2.1–2.4", "Sections two point one to two point four"),
    ("x = 3–5", "x equals three minus five"),
    # a tight fraction binds tighter than a spaced × ÷ * ·
    ("2/3 × 3/5", "two thirds times three fifths"),
    ("5/6 ÷ 1/3", "five sixths divided by one third"),
    ("3 × 1/2", "three times one half"),
    ("6 ÷ 1/2", "six divided by one half"),
    ("4 × 3/4", "four times three fourths"),
    ("2/3 * 3/5 = 2/5", "two thirds times three fifths equals two fifths"),
    ("1/2 · 4 = 2", "one half times four equals two"),
    ("3 x 1/2", "three times one half"),
    ("√(2/3 × 3/5)", "the square root of the quantity two thirds times three fifths"),
    # a question's "is" is not doubled
    ("Is 7 > 3?", "Is seven greater than three?"),
    ("Is 3 < 7 true?", "Is three less than seven true?"),
    ("Is x = 3?", "Is x equal to three?"),
    ("So, is x ≥ 4?", "So, is x greater than or equal to four?"),
    ("Does 3 + 4 = 7?", "Does three plus four equal seven?"),
    ("Is x = 3 or x = 4?", "Is x equal to three or x equal to four?"),
    ("Is x = 3? Or x = 4.", "Is x equal to three? Or x equals four."),
    ("The answer is x > 3.", "The answer is x is greater than three."),
    ("Is it true that 7 > 3?", "Is it true that seven is greater than three?"),
    ("What is 8-3?", "What is eight minus three?"),
    # function arguments and sets are not points; a relation in parentheses is not a quantity
    ("f(x, y)", "f of x, y"),
    ("g(1, 2)", "g of one, two"),
    ("x ∈ {1, 2}", "x is in the set one, two"),
    ("[2, 3]", "two, three"),
    ("Solve (3x + 5 = 20) now.", "Solve (three x plus five equals twenty) now."),
    ("(x = 3)", "(x equals three)"),
    # bare sqrt, decades, labels, spaced division
    ("sqrt 9", "the square root of nine"),
    ("sqrt x + 1", "the square root of x plus one"),
    ("Press the sqrt key.", "Press the square root key."),
    ("In the 1990s", "In the nineteen nineties"),
    ("the 1900s", "the nineteen hundreds"),
    ("the 2010s", "the twenty tens"),
    ("the 2000s", "the two thousands"),
    ("count by 100s", "count by hundreds"),
    ("count by 1000s", "count by thousands"),
    ("a 50-50 chance", "a fifty-fifty chance"),
    ("15-35 meters", "fifteen to thirty five meters"),
    ("Score: 3-2", "Score: three to two"),
    ("Vitamin B-12 helps.", "Vitamin B-twelve helps."),
    ("4 / 2 = 2", "four divided by two equals two"),
    ("10 / 2 = 5", "ten divided by two equals five"),
    ("x / 2", "x over two"),
    ("You got 7 / 10", "You got seven out of ten"),
    # an article before a negative number
    ("You get a -3 on the right.", "You get a negative three on the right."),
    ("Multiply by a -1.", "Multiply by a negative one."),
    ("A -5 is negative.", "A negative five is negative."),
    ("a - 3", "a minus three"),
    ("a-3", "a minus three"),
    ("x = a -3", "x equals a minus three"),
    ("Then a -3 = -1.", "Then a minus three equals negative one."),
    ("Let B-2 = 5", "Let B minus two equals five"),
    ("Do 2x + 3 = 11 next.", "Do two x plus three equals eleven next."),
    # ranges, labels and section numbers in prose
    ("Pick a number from 1-10.", "Pick a number from one to ten."),
    ("Grades 7-8", "Grades seven to eight"),
    ("It's 50-50.", "It's fifty-fifty."),
    ("Draw a 3-4-5 triangle.", "Draw a three-four-five triangle."),
    ("Section 2.3.1", "Section two point three point one"),
    ("Subtract 3 from 10-2.", "Subtract three from ten minus two."),
    ("10-3-2 = 5", "ten minus three minus two equals five"),
    ("Do 8-3 first.", "Do eight minus three first."),
]

# LaTeX that qwen2.5vl:3b wrote in a live run when not given the math system prompt.
LATEX_CASES: list[tuple[str, str]] = [
    (r"\( \frac{2}{3} \times \frac{3}{5} \)", "two thirds times three fifths"),
    (
        r"So, \( \frac{2 \times 3}{3 \times 5} = \frac{6}{15} \), which simplifies to \( \frac{2}{5} \).",
        "So, the quantity two times three, over the quantity three times five, equals six fifteenths, "
        "which simplifies to two fifths.",
    ),
    (r"\( \frac{5}{6} \div \frac{1}{3} \)", "five sixths divided by one third"),
    (r"$x^{2} + 1$", "x squared plus one"),
    (r"$$x = \frac{1}{2}$$", "x equals one half"),
    (r"\sqrt{16} = 4", "the square root of sixteen equals four"),
    (r"\sqrt[3]{8}", "the cube root of eight"),
    (r"\[ x \leq 5 \]", "x is less than or equal to five"),
    (r"\left( x + 1 \right)^2", "the quantity x plus one, squared"),
    (r"x_{1} + x_{2}", "x sub one plus x sub two"),
    (r"\frac{\frac{1}{2}}{3}", "one half over three"),
    (r"\dfrac{3}{4} of \text{the pizza}", "three fourths of the pizza"),
    (r"x \neq 3 \text{ and } y \geq -2", "x is not equal to three and y is greater than or equal to negative two"),
    ("It costs $5 and $10.", "It costs five dollars and ten dollars."),
    ("Pay $3 for $x$.", "Pay three dollars for x."),
    (r"$x$ and $y + 1$", "x and y plus one"),
    (r"3\text{ and }y", "three and y"),
]

# Every problem and answer in knowledge/math/quiz_bank.csv and worksheet_linear_equations.md, read by hand.
BANK_CASES: list[tuple[str, str]] = [
    ("1/4 + 2/4", "one fourth plus two fourths"),
    ("3/4", "three fourths"),
    ("1/2 + 1/3", "one half plus one third"),
    ("5/6", "five sixths"),
    ("3/4 - 1/2", "three fourths minus one half"),
    ("1/4", "one fourth"),
    ("2/5 + 3/10", "two fifths plus three tenths"),
    ("7/10", "seven tenths"),
    ("2/3 × 3/5", "two thirds times three fifths"),
    ("2/5", "two fifths"),
    ("5/6 ÷ 1/3", "five sixths divided by one third"),
    ("5/2", "five halves"),
    ("x + 7 = 12", "x plus seven equals twelve"),
    ("x = 5", "x equals five"),
    ("x - 4 = 9", "x minus four equals nine"),
    ("x = 13", "x equals thirteen"),
    ("5x = 35", "five x equals thirty five"),
    ("x = 7", "x equals seven"),
    ("x/3 = 6", "x over three equals six"),
    ("x = 18", "x equals eighteen"),
    ("y + 2.5 = 8", "y plus two point five equals eight"),
    ("y = 5.5", "y equals five point five"),
    ("-4x = 24", "negative four x equals twenty four"),
    ("x = -6", "x equals negative six"),
    ("3x + 5 = 20", "three x plus five equals twenty"),
    ("2x - 7 = 9", "two x minus seven equals nine"),
    ("x = 8", "x equals eight"),
    ("4x - 9 = 15", "four x minus nine equals fifteen"),
    ("x = 6", "x equals six"),
    ("x/4 + 3 = 8", "x over four plus three equals eight"),
    ("x = 20", "x equals twenty"),
    ("5x + 4 = -11", "five x plus four equals negative eleven"),
    ("x = -3", "x equals negative three"),
    ("7 - 2x = 1", "seven minus two x equals one"),
    ("x = 3", "x equals three"),
    ("2(x + 3) = 14", "two times the quantity x plus three, equals fourteen"),
    ("x = 4", "x equals four"),
    ("3(x - 2) = 15", "three times the quantity x minus two, equals fifteen"),
    ("4x + 2x - 5 = 13", "four x plus two x minus five equals thirteen"),
    ("5(x + 1) - 2x = 20", "five times the quantity x plus one, minus two x equals twenty"),
    ("2(3x - 4) = 4x + 6", "two times the quantity three x minus four, equals four x plus six"),
    ("3(x + 4) = 2(x + 9)", "three times the quantity x plus four, equals two times the quantity x plus nine"),
    ("x + y = 10 and x - y = 4", "x plus y equals ten and x minus y equals four"),
    ("x = 7 and y = 3", "x equals seven and y equals three"),
    ("y = 2x and x + y = 9", "y equals two x and x plus y equals nine"),
    ("x = 3 and y = 6", "x equals three and y equals six"),
    ("2x + y = 11 and x - y = 1", "two x plus y equals eleven and x minus y equals one"),
    ("x = 4 and y = 3", "x equals four and y equals three"),
    ("x^2 + 5x + 6 = 0", "x squared plus five x plus six equals zero"),
    ("x = -2 or x = -3", "x equals negative two or x equals negative three"),
    ("x^2 - 7x + 12 = 0", "x squared minus seven x plus twelve equals zero"),
    ("x = 3 or x = 4", "x equals three or x equals four"),
    ("x^2 - 9 = 0", "x squared minus nine equals zero"),
    ("x = 3 or x = -3", "x equals three or x equals negative three"),
    ("x^2 - 2x - 15 = 0", "x squared minus two x minus fifteen equals zero"),
    ("x = 5 or x = -3", "x equals five or x equals negative three"),
    # worksheet_linear_equations.md
    ("x + 9 = 15", "x plus nine equals fifteen"),
    ("x - 6 = 2", "x minus six equals two"),
    ("8x = 56", "eight x equals fifty six"),
    ("x/5 = 4", "x over five equals four"),
    ("2x + 3 = 11", "two x plus three equals eleven"),
    ("6x - 5 = 25", "six x minus five equals twenty five"),
    ("x/2 - 4 = 3", "x over two minus four equals three"),
    ("x = 14", "x equals fourteen"),
    ("10 - 3x = 4", "ten minus three x equals four"),
    ("x = 2", "x equals two"),
    ("4(x - 1) = 20", "four times the quantity x minus one, equals twenty"),
    ("3x + 2x + 4 = 29", "three x plus two x plus four equals twenty nine"),
    ("5x - 3 = 2x + 9", "five x minus three equals two x plus nine"),
    ("2(x + 5) = 3x + 4", "two times the quantity x plus five, equals three x plus four"),
    # how the worksheet lines read when the whole line is spoken
    ("1. x + 9 = 15", "one. x plus nine equals fifteen"),
    ("Answer: x = 6", "Answer: x equals six"),
    ("Problem 5: 2/3 × 3/5", "Problem five: two thirds times three fifths"),
]

ALL_CASES = CASES + REVIEW_CASES + LATEX_CASES + BANK_CASES
KNOWLEDGE = Path(__file__).resolve().parents[1] / "biobench_sar" / "knowledge" / "math"
DASHES = ["pages 3–5", "Try 2–3 more problems.", "5—6", "3 –5", "(x+1)–2", "3– 5", "x —2", "a–(b)", "1–2–3", "2—"]
DEEP_CHAINS = ["1+" * 3000 + "1", "1/" * 2000 + "2", "1-" * 2000 + "1", "2(x)" * 1500, "x" + " = x" * 1500]

PROSE = [
    "Great job! You solved it.",
    "Laguna de Bustillos is in Chihuahua (Mexico).",
    "Let's go step-by-step, it's a well-known trick.",
    "You're doing great — keep going.",
    "Remember: a negative times a negative is a positive.",
    "Do you have an exercise in mind, or should I make one up for you?",
    "Write the x-axis label (see above) first.",
    "e.g. this, i.e. that",
    "I – a teacher – said so.",
    "COVID",
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


@pytest.mark.parametrize(("text", "expected"), ALL_CASES)
def test_to_speech(text: str, expected: str) -> None:
    assert to_speech(text) == expected


def test_case_table_is_large() -> None:
    assert len(CASES) >= 60
    assert len(CORPUS) >= 30


@pytest.mark.parametrize("text", PROSE)
def test_prose_unchanged(text: str) -> None:
    assert to_speech(text) == text


@pytest.mark.parametrize("text", [t for t, _ in ALL_CASES] + CORPUS)
def test_idempotent(text: str) -> None:
    once = to_speech(text)
    assert to_speech(once) == once


@pytest.mark.parametrize("text", [t for t, _ in ALL_CASES] + CORPUS)
def test_single_pass_is_stable(text: str) -> None:
    once = mathspeak._convert(text)
    assert mathspeak._convert(once) == once


@pytest.mark.parametrize("text", CORPUS)
def test_corpus_has_no_forbidden_symbols(text: str) -> None:
    spoken = to_speech(text)
    assert not has_math_symbols(spoken), spoken
    assert not re.search(r"[\^/*=+<>%√×÷]|\d", spoken), spoken


@pytest.mark.parametrize("text", [t for t, _ in ALL_CASES])
def test_cases_have_no_forbidden_symbols(text: str) -> None:
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
    for text, _ in ALL_CASES:
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


def test_long_sign_and_power_chains_stay_in_band(caplog: pytest.LogCaptureFixture) -> None:
    signs = to_speech("-" * 3000 + "4")
    powers = to_speech("x" + "^x" * 3000)
    assert "four" in signs and "^" not in powers
    assert "falling back" not in caplog.text


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


def _bank_items() -> list[str]:
    items: list[str] = []
    with (KNOWLEDGE / "quiz_bank.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items += [row["problem"], row["answer"]]
    for line in (KNOWLEDGE / "worksheet_linear_equations.md").read_text(encoding="utf-8").splitlines():
        if m := re.match(r"\s*(?:\d+\.|Answer:)\s+(.+)", line):
            items.append(m.group(1))
    return items


def test_bank_cases_cover_the_knowledge_files() -> None:
    items = _bank_items()
    assert len(items) >= 80
    reviewed = {t for t, _ in BANK_CASES}
    assert [t for t in items if t not in reviewed] == []


@pytest.mark.parametrize("text", _bank_items())
def test_bank_items_read_without_digits_or_symbols(text: str) -> None:
    spoken = to_speech(text)
    assert not has_math_symbols(spoken), spoken
    assert not re.search(r"[\^/*=+<>%√×÷()]|\d|None", spoken), spoken
    assert to_speech(spoken) == spoken


@pytest.mark.parametrize("text", DASHES)
def test_dashes_never_say_none(text: str) -> None:
    spoken = to_speech(text)
    assert "None" not in spoken, spoken
    assert to_speech(spoken) == spoken


def test_tight_fraction_binds_tighter_than_spaced_division() -> None:
    toks = mathspeak._tokenize("6 ÷ 1/2")
    node = mathspeak._Parser(toks, 0).relation()
    assert node.kind == "bin" and node.text == "÷"
    assert node.kids[1].kind == "frac"


@pytest.mark.parametrize("text", DEEP_CHAINS, ids=["plus", "over", "minus", "implicit", "equals"])
def test_deep_chains_are_spoken_without_fallback(text: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    spoken = to_speech(text)
    assert caplog.records == []
    assert not re.search(r"[\d/+=()^-]", spoken), spoken[:80]
    assert to_speech(spoken) == spoken


def test_fallback_logs_one_line_and_spells_numbers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_: str) -> str:
        raise RecursionError("too deep")

    monkeypatch.setattr(mathspeak, "_convert", boom)
    caplog.set_level(logging.WARNING)
    out = to_speech("3x + 12 = 5")
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert caplog.records[0].exc_info is None
    assert not re.search(r"\d", out), out
    assert "twelve" in out


@pytest.mark.parametrize("text", ["x ∈ {1, 2}", "x ∉ A", r"\\frac{1}{2}", r"\\( x \\)", r"2 \\times 3"])
def test_set_membership_is_a_math_symbol(text: str) -> None:
    assert has_math_symbols(text)
    assert not has_math_symbols(to_speech(text))
