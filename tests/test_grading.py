from __future__ import annotations

import sys
from fractions import Fraction

import pytest

from biobench_sar import grading
from biobench_sar.grading import (
    compare,
    format_answers,
    grade,
    hint_for,
    incomplete,
    parse_answers,
    solution_sentences,
    solve_text,
    states_solution,
    to_math,
    without_solution,
)


def _plain(answers: frozenset[tuple[str, object]]) -> set[tuple[str, Fraction]]:
    return {(letter, Fraction(str(value))) for letter, value in answers}


# ---- parse_answers -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x = 5", {("x", 5)}),
        ("5", {("", 5)}),
        ("x=2.5", {("x", Fraction(5, 2))}),
        ("5/2", {("", Fraction(5, 2))}),
        ("x = 2 1/2", {("x", Fraction(5, 2))}),
        ("x = 2,5", {("x", Fraction(5, 2))}),  # a Spanish decimal comma
        ("x = -6", {("x", -6)}),
        ("x = −6", {("x", -6)}),  # a typographic minus
        ("y = 5.5", {("y", Fraction(11, 2))}),
        ("x = 5 or x = -3", {("x", 5), ("x", -3)}),
        ("x = 5 or -3", {("x", 5), ("x", -3)}),
        ("x = 3, x = -3", {("x", 3), ("x", -3)}),
        ("x = ±3", {("x", 3), ("x", -3)}),
        ("x = 5, y = 2", {("x", 5), ("y", 2)}),
        ("x=5,y=2", {("x", 5), ("y", 2)}),
        ("x = 7 and y = 3", {("x", 7), ("y", 3)}),
        ("(2, 3)", {("#1", 2), ("#2", 3)}),
        ("(x, y) = (7, 3)", {("#1", 7), ("#2", 3)}),
        ("x es igual a cinco", {("x", 5)}),
        ("x = 5 y x = -3", {("x", 5), ("x", -3)}),  # Spanish "y" between answers is "and", not the letter y
        ("x = 5 o x = menos tres", {("x", 5), ("x", -3)}),
        ("x vale siete", {("x", 7)}),
        ("la respuesta es cinco", {("", 5)}),
        ("x equals five", {("x", 5)}),
        ("x is negative three", {("x", -3)}),
        ("x equals two thirds", {("x", Fraction(2, 3))}),
        ("x = two and a half", {("x", Fraction(5, 2))}),
        ("x es igual a dos y medio", {("x", Fraction(5, 2))}),
        ("two point five", {("", Fraction(5, 2))}),
        ("treinta y cinco", {("", 35)}),
        ("the answer is 8.", {("", 8)}),
        ("Answer: x = 6", {("x", 6)}),
        ("Respuesta correcta: x = 6", {("x", 6)}),
        ("R: 7", {("", 7)}),
        ("A1. x = 4", {("x", 4)}),
        ("Sol. x = 6", {("x", 6)}),
        ("**x = 5**", {("x", 5)}),
        ("x = 5.", {("x", 5)}),
        ("x = 5 apples", {("x", 5)}),
        ("I got 5", {("", 5)}),
        ("x = 15/3 = 5", {("x", 5)}),
        ("3x + 5 = 20, x = 5", {("x", 5)}),  # the problem copied along: only the answer counts
        ("50%", {("", Fraction(1, 2))}),
    ],
)
def test_parse_answers(text: str, expected: set[tuple[str, object]]) -> None:
    assert _plain(parse_answers(text)) == {(letter, Fraction(value)) for letter, value in expected}  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    ["", None, "3x + 5 = 20", "3x = 15", "x = 2y", "x = 5x", "x = 2y + 1", "hello", "solve x", "x =", "1/0"],
)
def test_parse_answers_reads_no_answer(text: str | None) -> None:
    assert parse_answers(text) == frozenset()


def test_parse_answers_keeps_roots_exact() -> None:
    [(letter, value)] = parse_answers("x = √2")
    assert letter == "x" and abs(float(value) - 2**0.5) < 1e-12


# ---- grade -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("student", "expected", "verdict"),
    [
        ("x = 5", "x = 5", True),
        ("5", "x = 5", True),
        ("x=2.5", "5/2", True),
        ("x = 2 1/2", "5/2", True),
        ("2.5", "5/2", True),
        ("x = 4/6", "x = 2/3", True),
        ("y = 11/2", "y = 5.5", True),
        ("x = 7", "x = 5", False),
        ("x = -5", "x = 5", False),
        # several values: every one, in any order, and nothing else
        ("x = -3 or x = 5", "x = 5 or x = -3", True),
        ("x = 5 y x = -3", "x = 5 or x = -3", True),
        ("x = ±3", "x = 3 or x = -3", True),
        ("x = 5", "x = 5 or x = -3", False),
        ("x = 7 or x = -3", "x = 5 or x = -3", False),
        ("x = 5 or x = -3 or x = 1", "x = 5 or x = -3", False),
        # systems: each letter's value, a pair in alphabetical order
        ("x = 5, y = 2", "x = 5 and y = 2", True),
        ("y = 2, x = 5", "x = 5 and y = 2", True),
        ("(7, 3)", "x = 7 and y = 3", True),
        ("(3, 7)", "x = 7 and y = 3", False),
        ("x = 7", "x = 7 and y = 3", False),
        ("x = 5, y = 3", "x = 5", False),
        # words
        ("x es igual a cinco", "x = 5", True),
        ("x equals negative six", "x = -6", True),
        ("x equals seven", "x = 5", False),
        # nothing readable on one side: not graded
        ("7, 3", "x = 7 and y = 3", None),
        ("hello", "x = 5", None),
        ("x = 5", "a right angle", None),
        ("", "x = 5", None),
        ("x = 5", None, None),
    ],
)
def test_grade(student: str, expected: str | None, verdict: bool | None) -> None:
    assert grade(student, expected) is verdict


def test_compare_needs_both_sides() -> None:
    assert compare(frozenset(), parse_answers("x = 5")) is None
    assert compare(parse_answers("x = 5"), frozenset()) is None


def test_incomplete() -> None:
    expected = parse_answers("x = 5 or x = -3")
    assert incomplete(parse_answers("x = 5"), expected)
    assert not incomplete(parse_answers("x = 7"), expected), "a wrong value is not just missing ones"
    assert not incomplete(parse_answers("x = 5 or x = -3"), expected), "complete"
    assert incomplete(parse_answers("x = 7"), parse_answers("x = 7 and y = 3"))
    assert not incomplete(frozenset(), expected)


# ---- solve_text --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("problem", "answer"),
    [
        ("3x + 5 = 20", "x = 5"),
        ("Solve for x: 2(x + 5) = 3x + 4", "x = 6"),
        ("x/3 = 6", "x = 18"),
        ("-4x = 24", "x = -6"),
        ("y + 2.5 = 8", "y = 11/2"),
        ("7 - 2x = 1", "x = 3"),
        ("x^2 - 2x - 15 = 0", "x = 5 or x = -3"),
        ("x² = 49", "x = 7 or x = -7"),
        ("x^2 + 5x + 6 = 0", "x = -2 or x = -3"),
        ("x + y = 10 and x - y = 4", "x = 7 and y = 3"),
        ("y = 2x and x + y = 9", "x = 3 and y = 6"),
        ("x + y = 10 y x - y = 4", "x = 7 and y = 3"),  # Spanish "y" joining two equations
        ("2x + y = 11, x - y = 1", "x = 4 and y = 3"),
        ("1/2 + 1/3", "5/6"),
        ("5/6 ÷ 1/3", "5/2"),
        ("2/3 × 3/5", "2/5"),
        ("12 ÷ 4 = ?", "3"),
        ("6 x 7", "42"),
        ("What is 3 + 5?", "8"),
        ("3 x plus five equals twenty", "x = 5"),
    ],
)
def test_solve_text(problem: str, answer: str) -> None:
    solved = solve_text(problem)
    assert solved is not None and grade(format_answers(solved), answer) is True, format_answers(solved or frozenset())


@pytest.mark.parametrize(
    "problem",
    [
        "x = 5",  # already solved: nothing to solve
        "5 = x",
        "x^2 + 1 = 0",  # no real solution
        "y = 2x",  # one equation, two letters
        "how many apples are left?",
        "x^99 = 1",  # exponent too big to try
        "",
        None,
        "__import__('os').system('true')",  # never evaluated: our own parser
    ],
)
def test_solve_text_gives_none(problem: str | None) -> None:
    assert solve_text(problem) is None


def test_solved_form() -> None:
    assert grading.solved_form("x = 5") and grading.solved_form("x = 5 or x = -3")
    assert not grading.solved_form("3x + 5 = 20") and not grading.solved_form("hello")


def test_format_answers() -> None:
    assert format_answers(parse_answers("x = -3 or x = 5")) == "x = 5 or x = -3"
    assert format_answers(parse_answers("y = 3, x = 7")) == "x = 7 and y = 3"
    assert format_answers(parse_answers("5/6")) == "5/6"
    assert format_answers(frozenset()) is None


def test_without_sympy_plain_numbers_and_linear_problems_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    grading._sympy.cache_clear()
    monkeypatch.setitem(sys.modules, "sympy", None)  # import sympy -> ImportError
    try:
        assert grade("x = 5", "x = 5") is True and grade("x = 2.5", "5/2") is True
        assert grade("x = 7", "x = 5") is False
        assert grade("x = 5 or x = -3", "x = -3 or x = 5") is True
        assert format_answers(solve_text("3x + 5 = 20") or frozenset()) == "x = 5"
        assert solve_text("x^2 - 9 = 0") is None and hint_for("3x + 5 = 20") is None
        assert parse_answers("x = √2") == frozenset()
    finally:
        grading._sympy.cache_clear()


# ---- a reply that states the solution ----------------------------------------------------------------------------


SOLVED = solve_text("3x + 5 = 20")


@pytest.mark.parametrize(
    ("text", "states"),
    [
        ("So x = 5.", True),
        ("x is 5", True),
        ("x equals five.", True),
        ("The answer is five.", True),
        ("The value of x is 5.", True),
        ("x = 15/3.", True),
        ("You get x is 5 and you're done.", True),
        ("Three x equals fifteen. Now divide.", False),  # a step: "3x", not x
        ("Then 3x = 15.", False),
        ("Subtract 5 from both sides first.", False),  # the constant 5, not x = 5
        ("x plus five equals twenty.", False),  # the problem read out
        ("x equals seven.", False),  # not the solution
        ("-x = -5", False),
    ],
)
def test_states_solution_in_english(text: str, states: bool) -> None:
    assert states_solution(text, SOLVED, "en") is states


@pytest.mark.parametrize(
    ("text", "states"),
    [
        ("Entonces x es igual a cinco.", True),
        ("La respuesta es cinco.", True),
        ("x vale 5", True),
        ("x es igual a 5", True),
        ("Divide entre tres y x queda en cinco.", True),
        ("Resta cinco en ambos lados.", False),
        ("Tres x es igual a quince.", False),
        ("x es una letra.", False),
    ],
)
def test_states_solution_in_spanish(text: str, states: bool) -> None:
    assert states_solution(text, SOLVED, "es") is states


@pytest.mark.parametrize(
    ("text", "states"),
    [
        ("15 divided by 3 is 5.", True),
        ("Then divide 15 by 3 to get 5.", True),
        ("x turns out to be 5.", True),
        ("x ends up being 5.", True),
        ("That makes x five.", True),
        ("You should get five.", True),
        ("Fifteen divided by three gives you five.", True),
        ("x has to be five.", True),
        # hints: the step, the question, the numbers of a step
        ("Divide both sides by three.", False),
        ("What do you get when you divide fifteen by three?", False),
        ("Subtract five from both sides.", False),
        ("Subtract five from both sides to get 3x = 15.", False),
        ("Twenty minus five is fifteen.", False),
        ("Get the x term alone first.", False),
        ("What makes both sides equal?", False),
        ("You should get fifteen on the right.", False),
        ("y turns out to be 5.", False),  # another letter
    ],
)
def test_states_solution_said_another_way(text: str, states: bool) -> None:
    assert states_solution(text, SOLVED, "en") is states


@pytest.mark.parametrize(
    ("text", "states"),
    [
        ("x tiene que ser cinco.", True),
        ("Quince entre tres es cinco.", True),
        ("Obtienes cinco.", True),
        ("Te sale cinco.", True),
        ("Eso hace que x sea cinco.", True),
        ("Divide ambos lados entre tres.", False),
        ("¿Qué obtienes si divides quince entre tres?", False),
        ("Resta cinco en ambos lados.", False),
        ("Te queda tres x igual a quince.", False),
        ("Veinte menos cinco es quince.", False),
    ],
)
def test_states_solution_said_another_way_in_spanish(text: str, states: bool) -> None:
    assert states_solution(text, SOLVED, "es") is states


def test_states_the_roots_said_together() -> None:
    roots = solve_text("x^2 - 2x - 15 = 0")
    assert states_solution("The roots are 5 and -3.", roots)
    assert states_solution("Las raíces son cinco y menos tres.", roots, "es")
    assert not states_solution("The roots are where the graph crosses zero.", roots)
    assert not states_solution("Find two numbers that multiply to -15 and add to -2.", roots)
    assert not states_solution("Negative three times five is negative fifteen.", roots), "the numbers of a step"


def test_states_solution_of_quadratics_systems_and_arithmetic() -> None:
    roots = solve_text("x^2 - 2x - 15 = 0")
    assert states_solution("One answer is x = 5. Try factoring.", roots)
    assert states_solution("x equals negative three.", roots)
    system = solve_text("x + y = 10 and x - y = 4")
    assert states_solution("So x = 7. Now find y.", system) and not states_solution("Add them: 2x = 14.", system)
    total = solve_text("1/2 + 1/3")
    assert states_solution("One half plus one third equals five sixths.", total)
    assert states_solution("The answer is 5/6.", total)
    assert not states_solution("Find a common denominator: six.", total)
    assert not states_solution("x = 5", None) and not states_solution("x = 5", frozenset())


def test_solution_sentences_and_without_solution() -> None:
    text = "Subtract five from both sides. Then x equals five. Write it."
    assert solution_sentences(text, SOLVED, "en") == ["Then x equals five."]
    assert without_solution(text, SOLVED, "en") == "Subtract five from both sides. Write it."
    assert without_solution(text, None) == text


# ---- one hint for the next step ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("problem", "hint"),
    [
        ("3x + 5 = 20", "Subtract 5 from both sides first."),
        ("2x - 7 = 9", "Add 7 to both sides first."),
        ("7 - 2x = 1", "Subtract 7 from both sides first."),
        ("y + 2.5 = 8", "Subtract 2.5 from both sides first."),
        ("5x = 35", "Divide both sides by 5."),
        ("-4x = 24", "Divide both sides by -4."),
        ("x/3 = 6", "Multiply both sides by 3."),
        ("2(x + 3) = 14", "Start by multiplying out the brackets."),
        ("3x + 4 = x + 10", "Get all the x terms on one side first."),
        ("4x + 2x - 5 = 13", "Combine the x terms first."),
        ("x^2 - 2x - 15 = 0", "Factor it: look for two numbers that multiply to -15 and add to -2."),
        ("x^2 - 9 = 0", "Get x^2 alone, then take the square root of both sides. There are two answers."),
        ("y = 2x and x + y = 9", "Put 2x in place of y in the other equation."),
        ("x + y = 10 and x - y = 4", "Add or subtract the two equations so that one letter cancels out."),
        ("1/2 + 1/3", "Find a common denominator first. What number do 2 and 3 both go into?"),
        ("1/4 + 2/4", "The bottoms are the same, so work with the tops."),
        ("2/3 × 3/5", "Multiply the tops together and the bottoms together."),
        ("5/6 ÷ 1/3", "Flip the second fraction and multiply."),
    ],
)
def test_hint_for(problem: str, hint: str) -> None:
    assert hint_for(problem) == hint
    assert not states_solution(hint, solve_text(problem)), "a hint never gives the answer"


def test_hint_for_in_spanish() -> None:
    assert hint_for("3x + 5 = 20", "es") == "Primero resta 5 en ambos lados."
    assert hint_for("x^2 - 2x - 15 = 0", "es") == (
        "Factoriza: busca dos números que multiplicados den -15 y sumados den -2."
    )
    assert hint_for("y = 2x and x + y = 9", "es") == "Pon 2x en lugar de la y en la otra ecuación."


@pytest.mark.parametrize("problem", ["x = 5", "how many apples?", "12 + 4", "", None])
def test_hint_for_gives_none_when_code_cannot_tell(problem: str | None) -> None:
    assert hint_for(problem) is None


def test_to_math() -> None:
    assert to_math("three x plus five equals twenty") == "3 x + 5 = 20"
    assert to_math("Tres x más cinco es igual a veinte", "es") == "3 x + 5 = 20"
    assert to_math("x = 2y") == "x = 2y", "a term keeps its spacing"
    assert to_math("x es una variable", "es") == "x = una variable", "un / una is no number before a noun"
    assert to_math("un medio", "es") == "1/2"


def test_an_answer_written_without_its_letter_is_still_the_letters_value() -> None:
    bare = parse_answers("8")  # a worksheet's "The answer is 8." for 2x = 16
    assert states_solution("Not quite, x = 8.", bare) and states_solution("So y is eight.", bare)
    assert not states_solution("Divide both sides by 2.", bare)
    pair = parse_answers("(7, 3)")
    assert states_solution("So x = 7.", pair) and not states_solution("Add the two equations.", pair)
    named = parse_answers("x = 5")
    assert not states_solution("y = 5", named), "another letter of a named answer is not its value"


# ---- arithmetic not worked out yet is no answer ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "1/2 + 1/3",
        "3/4 - 1/2",
        "12 ÷ 4",
        "2/3 × 3/4",
        "0.5 + 0.25",
        "1/5 + 2/5",
        "2/6 + 3/6",  # worked part of the way for 1/2 + 1/3
        "1/2 + 1/3 = 2/6 + 3/6",
        "12 ÷ 4 = ?",
        "1/2 + 1/3 =",
        "2^3",
        "one half plus one third",
        "un medio más un tercio",
    ],
)
def test_arithmetic_not_worked_out_is_no_answer(text: str) -> None:
    assert parse_answers(text) == frozenset(), "worked out in code it is its own answer: never graded as one"
    assert grading.unfinished(text)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1/2 + 1/3 = 5/6", Fraction(5, 6)),  # the value after the last "=", no letter on the left
        ("2/6 + 3/6 = 5/6", Fraction(5, 6)),
        ("1/2 + 1/3 = 2/6 + 3/6 = 5/6", Fraction(5, 6)),
        ("12 ÷ 4 = 3", 3),
        ("5/6", Fraction(5, 6)),
        ("0.75", Fraction(3, 4)),
        ("1 1/2", Fraction(3, 2)),
        ("-3", -3),
        ("10/12", Fraction(5, 6)),
    ],
)
def test_worked_arithmetic_and_single_values_are_answers(text: str, value: object) -> None:
    assert _plain(parse_answers(text)) == {("", Fraction(str(value)))}
    assert not grading.unfinished(text)


def test_letters_and_exact_roots_are_not_unfinished() -> None:
    for text in ("3x + 5 = 20", "3x = 15", "x = 5", "x = 15/3", "x = 1 + √2", "", None, "hello"):
        assert not grading.unfinished(text), text
    assert grade("x = 15/3", "x = 5") is True and grade("x = 5", "x = 5") is True
    assert len(parse_answers("x = 1 + √2 or 1 - √2")) == 2, "an exact root with a + is a value, not unworked"
    assert grade("1/2 + 1/3 = 5/6", "5/6") is True and grade("1/2 + 1/3 = 2/5", "5/6") is False
    assert grade("1/2 + 1/3", "5/6") is None, "the problem itself is not graded"


def test_scratch_arithmetic_next_to_a_step_is_not_the_answer() -> None:
    assert _plain(parse_answers("20 - 5 = 15\nx = 5")) == {("x", 5)}
    assert parse_answers("20 - 5 = 15\n3x = 15") == frozenset()
    assert _plain(parse_answers("15 - 9 = 6, so x = 6")) == {("x", 6)}


def test_same_problem() -> None:
    assert grading.same_problem("1/2+1/3", "1/2 + 1/3")
    assert grading.same_problem("6/8", "Simplify 6/8") and grading.same_problem("12 ÷ 4", "12 ÷ 4 = ?")
    assert grading.same_problem("one half plus one third", "1/2 + 1/3")
    assert not grading.same_problem("5/6", "1/2 + 1/3") and not grading.same_problem("6/8", "3/4")
    assert not grading.same_problem("", "1/2 + 1/3") and not grading.same_problem("5", None)


# ---- a check or a reason after an answer -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x = 5 (check: 3(5) + 5 = 20)", {("x", 5)}),
        ("x = 4 because 2 times 4 is 8", {("x", 4)}),
        ("x = 6: subtract 9 from both sides, then divide by 2", {("x", 6)}),
        ("x = 6 - subtract 9 from both sides", {("x", 6)}),
        ("x = 6 — subtract nine, then divide by two", {("x", 6)}),
        ("x = 6; divide both sides by 2", {("x", 6)}),
        ("x = 5 since 15/3 = 5", {("x", 5)}),
        ("x = 5, check: 3(5) + 5 = 20", {("x", 5)}),
        ("x = 6 (subtract nine first)", {("x", 6)}),
        ("x = 7 because 7 + 3 = 10, y = 3 because 7 - 3 = 4", {("x", 7), ("y", 3)}),
        ("x = 5 or x = -3 (both make it 0)", {("x", 5), ("x", -3)}),
        ("x = 4 porque 2 por 4 es 8", {("x", 4)}),
        ("x = 4 ya que 2 por 4 es 8", {("x", 4)}),
        ("x = 5 (comprobación: 3(5) + 5 = 20)", {("x", 5)}),
        ("5/6 (common denominator 6)", {("", Fraction(5, 6))}),
        # kept: answers, pairs, letters, labels
        ("x = 5; y = 2", {("x", 5), ("y", 2)}),
        ("x = 5 y y = 2", {("x", 5), ("y", 2)}),
        ("x = 5 o x = -3", {("x", 5), ("x", -3)}),
        ("(2, 3)", {("#1", 2), ("#2", 3)}),
        ("A1: x = 4", {("x", 4)}),
        ("check: x = 5", {("x", 5)}),
    ],
)
def test_explanations_after_an_answer_are_cut(text: str, expected: set[tuple[str, object]]) -> None:
    assert _plain(parse_answers(text)) == {(letter, Fraction(str(value))) for letter, value in expected}


def test_an_answer_key_with_a_check_grades_its_value() -> None:
    key = "x = 5 (check: 3(5) + 5 = 20)"
    assert grade("x = 5", key) is True and grade("x = 20", key) is False
    because = "x = 4 because 2 times 4 is 8"
    assert grade("x = 4", because) is True and grade("x = 8", because) is False
    assert grade("x = 6", "x = 6: subtract 9 from both sides, then divide by 2") is True
    assert grade(because, "x = 4") is True, "a student's reason is cut too"


# ---- answers on several lines or side by side --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x = 5\nx = -3", {("x", 5), ("x", -3)}),
        ("x = 5 x = -3", {("x", 5), ("x", -3)}),
        ("x = 7 y = 3", {("x", 7), ("y", 3)}),
        ("x = 7\ny = 3", {("x", 7), ("y", 3)}),
        ("x^2 - 2x - 15 = 0\nx = 5\nx = -3", {("x", 5), ("x", -3)}),
        ("x = 5 ; x = -3", {("x", 5), ("x", -3)}),
        ("3x + 5 = 20 x = 5", {("x", 5)}),
        ("x + y = 10\nx - y = 4\nx = 7, y = 3", {("x", 7), ("y", 3)}),
    ],
)
def test_answers_on_several_lines_or_side_by_side(text: str, expected: set[tuple[str, object]]) -> None:
    assert _plain(parse_answers(text)) == {(letter, Fraction(str(value))) for letter, value in expected}


def test_several_lines_grade_as_one_answer() -> None:
    assert grade("x = 5\nx = -3", "x = 5 or x = -3") is True
    assert grade("x = 5 x = -3", "x = 5 or x = -3") is True
    assert grade("x = 7 y = 3", "x = 7 and y = 3") is True
    assert grade("x = 5\nx = 3", "x = 5 or x = -3") is False
    for step in ("2 x = 10", "two x equals ten", "3x = 15"):
        assert parse_answers(step) == frozenset(), "a step is still no answer"


def test_to_math_keeps_line_breaks() -> None:
    assert to_math("x  =  5\n\n x = -3 ") == "x = 5\nx = -3"
    assert solve_text("x + y = 10\nx - y = 4") == solve_text("x + y = 10 and x - y = 4")
