from __future__ import annotations

import sys
import threading
import time
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


@pytest.mark.parametrize(
    ("board", "solution"),
    [
        ("1/2 + 1/3 = 2/5", "5/6"),  # the student's own wrong result copied with the problem
        ("1/2 + 1/3 = 3/6 + 2/6 = 5/6", "5/6"),
        ("1/2 + 1/3\n5/6", "5/6"),
        ("un medio más un tercio es igual a dos quintos", "5/6"),
        ("12 ÷ 4 = 2", "3"),
        ("2x + 3 = 11\nx = 5", "x = 4"),  # a wrong answer below the problem
        ("2x + 3 = 11 ; 2x = 14", "x = 4"),  # a wrong step
        ("x + y = 10 ; x - y = 4 ; x = 8", "x = 7 and y = 3"),
        ("x^2 = 49 ; x = 7", "x = 7 or x = -7"),
        ("x = 5 ; 2x + 3 = 11", "x = 4"),
        ("3x + 5 = 20", "x = 5"),
    ],
)
def test_board_solution_is_the_problems_whatever_is_copied_with_it(board: str, solution: str) -> None:
    assert format_answers(grading.board_solution(board) or frozenset()) == solution


def test_board_solution_of_a_board_with_no_problem_is_none() -> None:
    for board in ("x = 5", "x = 5 or x = -3", "", None, "hello", "= 5/6"):
        assert grading.board_solution(board) is None, board


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


@pytest.mark.parametrize(
    ("text", "problem", "language"),
    [
        ("Five sixths is the answer.", "1/2 + 1/3", "en"),
        ("It is actually five-sixths.", "1/2 + 1/3", "en"),
        ("So it is four.", "2x + 3 = 11", "en"),
        ("It's three.", "12 ÷ 4", "en"),
        ("The answer is twenty-five.", "x - 5 = 20", "en"),
        ("Four is the answer.", "2x + 3 = 11", "en"),
        ("Te debe dar cinco sextos.", "1/2 + 1/3", "es"),
        ("Debería dar 5/6.", "1/2 + 1/3", "es"),
        ("Tres sextos más dos sextos son cinco sextos.", "1/2 + 1/3", "es"),
        ("Son cinco sextos.", "1/2 + 1/3", "es"),
        ("Eso es cuatro.", "2x + 3 = 11", "es"),
        ("Es cuatro.", "2x + 3 = 11", "es"),
    ],
)
def test_states_solution_said_as_what_it_comes_to(text: str, problem: str, language: str) -> None:
    # These leaked on tutor-core and in round one: the value before "is the answer", a pronoun's value, "debe dar".
    assert states_solution(text, solve_text(problem), language), text


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("It is a good idea to subtract 3 first.", "en"),
        ("It is 11 on the right side.", "en"),
        ("So it is 2x = 8 now.", "en"),
        ("It's 4 times x.", "en"),
        ("Is it four? Check it.", "en"),
        ("4x is the answer we want to undo.", "en"),
        ("What do you get when you divide eight by two?", "en"),
        ("Son dos pasos.", "es"),
        ("Debe quedar la x sola.", "es"),
        ("Te debe dar la x sola.", "es"),
    ],
)
def test_a_hint_that_only_ends_near_the_solution_is_not_flagged(text: str, language: str) -> None:
    for problem in ("2x + 3 = 11", "x + 2 = 4"):
        assert not states_solution(text, solve_text(problem), language), (text, problem)


def test_hyphenated_number_words_are_one_number() -> None:
    assert to_math("twenty-five") == "25" and to_math("five-sixths") == "5/6" and to_math("one-half") == "1/2"
    assert to_math("x-y") == "x-y" and to_math("well-known") == "well-known"


def test_board_solution_of_a_board_with_an_arrow() -> None:
    for board in ("2x + 3 = 11 → x = 5", "2x + 3 = 11 -> x = 5", "2x + 3 = 11 => x = 4", "2x + 3 = 11 ⇒ 2x = 8"):
        assert grading.board_solution(board) == solve_text("2x + 3 = 11"), board


def test_board_problem_is_the_part_that_board_solution_solves() -> None:
    assert grading.board_problem("1/2 + 1/3 = 2/5") == "1/2 + 1/3"
    assert grading.board_problem("2x + 3 = 11\nx = 5") == "2x + 3 = 11"
    assert grading.board_problem("x + y = 10 ; x - y = 4 ; x = 8") == "x + y = 10 ; x - y = 4"
    assert grading.board_problem("x = 5") is None and grading.board_problem("") is None


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


@pytest.mark.parametrize(
    ("problem", "answer", "check"),
    [
        ("3x + 5 = 20", "x = 5", "3(5) + 5 = 20"),
        ("3x + 5 = 20", "x = 5", "3 * 5 + 5 = 20"),
        ("3x + 5 = 20", "x = 5", "15 + 5 = 20"),
        ("2x - 7 = 9", "x = 8", "2(8) - 7 = 9"),
        ("x + 7 = 12", "x = 5", "5 + 7 = 12"),
        ("5x + 4 = -11", "x = -3", "5(-3) + 4 = -11"),
        ("3x + 5 = 20", "x = 4", "3(4) + 5 = 17"),  # the check of a wrong answer is no answer either
        ("3x + 5 = 20", "x = 4", "3(4) + 5 = 20"),
        ("12 ÷ 4", "3", "3 × 4 = 12"),
        ("1/2 + 1/3", "5/6", "5/6 - 1/3 = 1/2"),
    ],
)
def test_a_substitution_check_on_the_bottom_line_is_never_the_answer(problem: str, answer: str, check: str) -> None:
    assert parse_answers(check, problem) == frozenset(), "alone, a check line is no answer"
    assert not grading.unfinished(check), "nor unfinished work"
    assert parse_answers(f"{answer} ; {check}", problem) == parse_answers(answer), "the answer line counts"
    assert parse_answers(f"{answer}\n{check}", problem) == parse_answers(answer)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3 = x", {("x", 3)}),
        ("3=x", {("x", 3)}),
        ("6 = 2x\n3 = x", {("x", 3)}),  # 7 - 2x = 1 worked out
        ("6 = 2x ; 3 = x", {("x", 3)}),
        ("7 = x", {("x", 7)}),
        ("-3 = x", {("x", -3)}),
        ("15/3 = x", {("x", 5)}),
        ("3 = y", {("y", 3)}),
        ("x = 7 ; 3 = y", {("x", 7), ("y", 3)}),
        ("tres es igual a x", {("x", 3)}),
    ],
)
def test_a_finished_answer_with_the_letter_on_the_right_grades_like_x_equals(
    text: str, expected: set[tuple[str, object]]
) -> None:
    assert _plain(parse_answers(text)) == {(letter, Fraction(str(value))) for letter, value in expected}
    assert not grading.unfinished(text), "a finished answer is never 'not worked out yet'"


@pytest.mark.parametrize("tail", [" ✓", " ✔", " ok", " :)", " □", " ∎", " <-", " answer", " final", " ✅", " sí"])
def test_the_letter_on_the_right_with_a_mark_after_it_grades_like_x_equals(tail: str) -> None:
    for text, value in (("3 = x", 3), ("-3 = x", -3), ("6 = 2x\n3 = x", 3), ("15/3 = x", 5)):
        assert _plain(parse_answers(text + tail)) == {("x", value)}, text + tail
    for step in ("3 = x + 1", "3 = xy", "3 = x 2", "3 = x^2", "3 = x(2)", "3 = x - 1", "3 = 2x", "x = 2y"):
        assert parse_answers(step + tail) == frozenset(), step + tail
    assert _plain(parse_answers("x = 5cm" + tail)) == {("x", 5)}, "a unit touching the value is no letter"


def test_an_answer_with_the_letter_on_the_right_and_its_check_side_by_side() -> None:
    assert _plain(parse_answers("3 = x 7 - 2(3) = 1", "7 - 2x = 1")) == {("x", 3)}
    assert _plain(parse_answers("3 = x ; 7 - 2(3) = 1", "7 - 2x = 1")) == {("x", 3)}


def test_subscripted_roots_are_values_of_their_letter() -> None:
    assert grade("x₁ = 3, x₂ = 4", "x = 3 or x = 4") is True
    assert grade("x_1 = 3 ; x_2 = 4", "x = 3 or x = 4") is True
    assert grade("x₁ = 3, x₂ = 5", "x = 3 or x = 4") is False
    assert grade("x1 = 3, x2 = 4", "x = 3 or x = 4") is True and grade("x1 = 3 x2 = 4", "x = 3 or x = 4") is True
    assert parse_answers("3x2 = 6") == frozenset() and parse_answers("2x1 = 4") == frozenset(), "a product, a step"


def test_a_step_with_the_letter_on_the_right_is_still_no_answer() -> None:
    for step in ("6 = 2x", "2 = 2x", "x + 3 = x", "20 = 3x + 5"):
        assert parse_answers(step) == frozenset(), step
        assert not grading.unfinished(step), f"{step}: a step, not unfinished arithmetic"
    assert grade("3 = x", "x = 3") is True and grade("4 = x", "x = 3") is False
    assert grade("6 = 2x\n3 = x", "x = 3") is True


def test_the_arithmetic_problem_worked_out_is_still_its_answer() -> None:
    for read in ("1/2 + 1/3 = 5/6", "3/6 + 2/6 = 5/6", "1/2 + 1/3 = 3/6 + 2/6 = 5/6"):
        assert _plain(parse_answers(read, "1/2 + 1/3")) == {("", Fraction(5, 6))}, read
    assert _plain(parse_answers("1/2 + 1/3 = 2/5", "1/2 + 1/3")) == {("", Fraction(2, 5))}, "wrong, but an answer"
    assert _plain(parse_answers("12 ÷ 4 = 3", "12 ÷ 4 = ?")) == {("", 3)}
    assert _plain(parse_answers("15 + 5 = 20")) == {("", 20)}, "without the problem, arithmetic worked out as before"
    word = "A rectangle is 3 cm wide and 4 cm long. Find its area."
    assert _plain(parse_answers("3 × 4 = 12", word)) == {("", 12)}, "a word problem's arithmetic is its answer"


WORDED_CHECKS = [
    ("Despeja x: 3x + 5 = 20", "x = 5", "3(5) + 5 = 20"),
    ("2x - 7 = 9. Find x.", "x = 8", "2(8) - 7 = 9"),
    ("Encuentra el valor de x: 5x + 4 = -11", "x = -3", "5(-3) + 4 = -11"),
    ("Resuelve la ecuación 7 - 2x = 1", "x = 3", "7 - 2(3) = 1"),
    ("3x + 5 = 20. Find x.", "x = 5", "3(5) + 5 = 20"),
    ("Halla x en 3x + 5 = 20", "x = 5", "15 + 5 = 20"),
    ("Solve for x in the equation 2x + 3 = 11", "x = 4", "2(4) + 3 = 11"),
    ("A number doubled plus 3 is 11. What is the number?", "n = 4", "2(4) + 3 = 11"),  # a word problem, key with n
]


@pytest.mark.parametrize(("problem", "answer", "check"), WORDED_CHECKS)
def test_a_check_under_a_worded_problem_with_an_answer_key_is_never_the_answer(
    problem: str, answer: str, check: str
) -> None:
    # Worksheet problems are stored as written; tutor-core never graded these checks, round one graded them wrong.
    expected = parse_answers(answer)
    assert parse_answers(check, problem, expected) == frozenset(), "alone, the check is no answer"
    assert grading.only_checks(check, problem, expected), "so the check asks again for the answer line"
    assert parse_answers(f"{answer} ; {check}", problem, expected) == expected
    assert parse_answers(f"x = 99 ; {check}", problem, expected) == parse_answers("x = 99"), "a wrong answer counts"


def test_the_letter_of_a_worded_problem_is_found_without_its_answer_key() -> None:
    for problem, _, check in WORDED_CHECKS[:-1]:
        assert parse_answers(check, problem) == frozenset(), problem
    assert grading.only_checks("3(5) + 5 = 20", "Despeja x: 3x + 5 = 20", parse_answers("5")), "a bare key too"


def test_arithmetic_under_a_word_problem_counts_only_when_it_comes_to_the_answer() -> None:
    area = "A rectangle is 3 cm wide and 4 cm long. Find its area."
    assert _plain(parse_answers("3 × 4 = 12", area, parse_answers("12"))) == {("", 12)}
    assert parse_answers("3 + 4 = 7", area, parse_answers("12")) == frozenset(), "true, not the answer: a check"
    pizza = "Ana eats half a pizza and then a third of one. How much did she eat?"
    assert _plain(parse_answers("1/2 + 1/3 = 2/5", pizza, parse_answers("5/6"))) == {("", Fraction(2, 5))}, "false"
    assert _plain(parse_answers("5/6", pizza, parse_answers("5/6"))) == {("", Fraction(5, 6))}


MARKS = [" ✓", " ✔", " ✅", " √", " true", " yes", " ok", " correct", " sí", " →", " ->", " :)", " <-"]


@pytest.mark.parametrize("mark", MARKS)
def test_a_check_line_with_a_tick_or_a_word_after_it_is_still_a_check(mark: str) -> None:
    for check in ("3(5) + 5 = 20", "3(5)+5=20", "15 + 5 = 20", "20 = 20"):
        line = check + mark
        assert parse_answers(line, "3x + 5 = 20") == frozenset(), line
        assert grading.only_checks(line, "3x + 5 = 20"), line
        assert parse_answers(f"x = 5 ; {line}", "3x + 5 = 20") == parse_answers("x = 5")
    assert _plain(parse_answers("1/2 + 1/3 = 5/6" + mark, "1/2 + 1/3")) == {("", Fraction(5, 6))}, "an answer"
    assert _plain(parse_answers("1/2 + 1/3 = 2/5" + mark, "1/2 + 1/3")) == {("", Fraction(2, 5))}


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1/2 + 1/3 ; = 3/6 + 2/6 ; = 5/6", Fraction(5, 6)),
        ("1/2 + 1/3\n= 3/6 + 2/6\n= 5/6", Fraction(5, 6)),
        ("1/2 + 1/3\n= 2/5", Fraction(2, 5)),
        ("= 5/6", Fraction(5, 6)),
        ("12 ÷ 4\n= 3", Fraction(3)),
    ],
)
def test_a_vertical_chain_is_read_down_to_its_last_line(text: str, value: Fraction) -> None:
    problem = "12 ÷ 4" if "÷" in text else "1/2 + 1/3"
    assert _plain(parse_answers(text, problem)) == {("", value)}
    assert not grading.unfinished(text)
    assert grading.unfinished("1/2 + 1/3\n= 3/6 + 2/6"), "not at its bottom yet"
    assert parse_answers("x = 5\n3(5) + 5\n= 15 + 5\n= 20", "3x + 5 = 20") == parse_answers("x = 5"), "a check"


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x = 5 (3(5) + 5 = 20)", {("x", 5)}),
        ("x = 3 (7 - 2(3) = 1)", {("x", 3)}),
        ("x = 5 [3(5) + 5 = 20]", {("x", 5)}),
        ("x = 5 3(5) + 5 = 20", {("x", 5)}),
        ("x=5 3(5)+5=20", {("x", 5)}),
        ("x = 3 7 - 2(3) = 1", {("x", 3)}),
        ("x = -3 5(-3) + 4 = -11", {("x", -3)}),
        ("x = 1/2 2(1/2) + 1 = 2", {("x", Fraction(1, 2))}),
        # kept: a mixed number, a chain, a product, letters in brackets
        ("x = 2 1/2 = 5/2", {("x", Fraction(5, 2))}),
        ("x = 15 - 5 = 10", {("x", 10)}),
        ("x = 2 (3 + 1) = 8", {("x", 8)}),
        ("(x = 7, y = 3)", {("x", 7), ("y", 3)}),
    ],
)
def test_a_plain_math_check_in_brackets_or_beside_the_answer_is_cut(
    text: str, expected: set[tuple[str, object]]
) -> None:
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


# ---- nothing takes long ------------------------------------------------------------------------------------------

TOWER = "x = " + "(" * 9 + "9" + ")^6" * 9  # 9^(6^9) worked out exactly: about ten million digits
BIG_BASE_TOWER = "x = " + "(" * 7 + "9" * 40 + ")^6" * 7
POLY_TOWER = "(" * 4 + "x + 1" + ")^6" * 4 + " = 0"  # degree 1296 multiplied out


def test_a_short_power_tower_never_freezes_the_process() -> None:
    from biobench_sar.llm import vlm_client as vc
    from biobench_sar.schemas import AvatarAction

    task = {"kind": "quiz", "stage": "check", "number": 1, "total": 3, "problem": "3x + 5 = 20", "answer": "x = 5"}
    praise = AvatarAction("Correct, great job!", "celebrating", 1, 1)
    vc.check_quiz_answer(praise, task, "x = 5")  # sympy and the parser loaded before the clock starts
    gaps: list[float] = []
    done = threading.Event()

    def tick() -> None:  # the camera loop's 30 Hz
        last = time.perf_counter()
        while not done.is_set():
            time.sleep(1 / 30)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    time.sleep(0.1)
    try:
        read = [parse_answers(text) for text in (TOWER, BIG_BASE_TOWER)]
        solved = [solve_text(text) for text in (POLY_TOWER, TOWER)]
        checked = [vc.check_quiz_answer(praise, task, text) for text in (TOWER, BIG_BASE_TOWER)]
        guarded = [vc.solutions_for("help", None, text) for text in (POLY_TOWER, TOWER)]
        hinted = hint_for(POLY_TOWER)
    finally:
        done.set()
        ticker.join()
    assert max(gaps) < 0.25, f"the 30 Hz ticker stalled for {max(gaps):.2f} s"
    assert read == [frozenset(), frozenset()] and solved == [None, None] and guarded == [None, None]
    assert all(a.verdict is None for a in checked) and hinted is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "text",
    [
        "x = " + "(" * 4 + "9" + ")^6" * 4,  # 9^1296: past MAX_BITS before it is worked out
        "x = " + "(" * 9 + "1/9" + ")^6" * 9,
        "x = " + "(" * 9 + "9" + ")^-6" * 9,
        "x = " + "(" * 6 + "√2 + √3" + ")^6" * 6,
        "x = √(" + "*".join(["(" + "9" * 50 + ")^6"] * 4) + ")",  # sympy looked for square factors for seconds
    ],
)
def test_powers_and_roots_too_big_to_work_out_are_no_value(text: str) -> None:
    assert parse_answers(text) == frozenset()


def test_powers_a_problem_needs_still_work() -> None:
    assert _plain(parse_answers("x = 2^3")) == {("x", 8)} and _plain(parse_answers("x = √(2^6)")) == {("x", 8)}
    assert _plain(parse_answers("x = ((9)^6)^6")) == {("x", 9**36)}, "300 digits is still a number"
    assert format_answers(solve_text("(x + 1)^2 = 9") or frozenset()) == "x = 2 or x = -4"
    assert format_answers(solve_text("(2x - 1)^2 = 25") or frozenset()) == "x = 3 or x = -2"


def test_only_what_solves_fast_is_solved() -> None:
    assert format_answers(solve_text("x^4 - 5x^2 + 4 = 0") or frozenset()) == "x = 2 or x = 1 or x = -1 or x = -2"
    assert format_answers(solve_text("x^3 = 8") or frozenset()) == "x = 2", "it factors: one real root"
    assert format_answers(solve_text("x^3 - 6x^2 + 11x - 6 = 0") or frozenset()) == "x = 3 or x = 2 or x = 1"
    for slow in (
        "3x^4 - 7x^3 + x - 11 = 0",  # no factors: roots in radicals took a third of a second
        "2x^3 - 3x + 7 = 0",
        "3x^2 + 5y^2 - 7xy = 11 and 2xy + 3y = 13",  # two quadratics: four seconds
        "x^4 + y^4 = 17 and x^3 y = 8",
        "√2 x^2 + √3 y = √5 and √7 x - y = 1",  # one quadratic, root coefficients: four seconds
        "x^2 + 3y = 7 and 2x - y^2 = 5",
        "(x+y+z+1)^6 * (x+y+z+2)^6 = 0",
        POLY_TOWER,
    ):
        assert solve_text(slow) is None, slow
    assert format_answers(solve_text("x + y = 10 and x - y = 4") or frozenset()) == "x = 7 and y = 3"


@pytest.mark.parametrize(
    ("problem", "solution"),
    [
        ("y = x^2 and x + y = 2", "x = 1 and x = -2 and y = 4 and y = 1"),
        ("x^2 + y^2 = 25 and x - y = 1", "x = 4 and x = -3 and y = 3 and y = -4"),
        ("√2 x + y = 3 and x - y = 1", "x = -4 + 4√(2) and y = -5 + 4√(2)"),
        ("x^3 = 2", "x = 2^(1/3)"),
        ("x^4 = 7", "x = 7^(1/4) or x = -7^(1/4)"),
    ],
)
def test_a_system_with_one_quadratic_a_root_coefficient_or_a_power_still_solves(problem: str, solution: str) -> None:
    # tutor-core solved these in 3-39 ms; the round-one gate (linear systems only, a root counted as a letter,
    # cubics only when they factor) gave None, so the help guard was off and a keyless quiz problem never graded.
    solve_text("x + y = 10 and x - y = 4")  # sympy warmed up before the clock starts
    start = time.perf_counter()
    solved = solve_text(problem)
    assert time.perf_counter() - start < 0.5, "still fast"
    assert format_answers(solved or frozenset()) == solution
    assert grading.board_solution(problem) == solved
