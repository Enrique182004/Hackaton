from __future__ import annotations

import re

import pytest

from biobench_sar.llm.mathspeak_es import fraction_to_words_es, has_math_symbols, number_to_words_es, to_speech_es

FORBIDDEN = re.compile(r"[\d^/*=+<>×·÷√∛%≤≥≠≈±²³½¼¾π\ue000-\uf8ff]")

CASES = [
    # contract examples
    ("2/3", "dos tercios"),
    ("1/2", "un medio"),
    ("3/4", "tres cuartos"),
    ("5/8", "cinco octavos"),
    ("7/11", "siete onceavos"),
    ("x/2", "x sobre dos"),
    ("x^2", "x al cuadrado"),
    ("x^3", "x al cubo"),
    ("x^n", "x a la n"),
    ("3x+5=12", "tres x más cinco es igual a doce"),
    ("-4", "menos cuatro"),
    ("9 - 4", "nueve menos cuatro"),
    ("x = -4", "x es igual a menos cuatro"),
    ("0.5", "cero punto cinco"),
    ("25%", "veinticinco por ciento"),
    ("√9", "la raíz cuadrada de nueve"),
    ("2*3", "dos por tres"),
    ("2 × 3", "dos por tres"),
    ("2·3", "dos por tres"),
    ("6 ÷ 2", "seis entre dos"),
    ("x < 5", "x es menor que cinco"),
    ("x > 5", "x es mayor que cinco"),
    ("x ≤ 5", "x es menor o igual que cinco"),
    ("x ≥ 5", "x es mayor o igual que cinco"),
    ("x ≠ 3", "x es diferente de tres"),
    ("2 1/2", "dos y un medio"),
    ("2(x+1)", "dos por la cantidad x más uno"),
    ("(2, 3)", "el punto dos, tres"),
    ("21", "veintiuno"),
    ("32", "treinta y dos"),
    ("100", "cien"),
    ("150", "ciento cincuenta"),
    ("500", "quinientos"),
    ("1250", "mil doscientos cincuenta"),
    # ASCII and unicode spellings
    ("x <= 5", "x es menor o igual que cinco"),
    ("x >= 5", "x es mayor o igual que cinco"),
    ("x != 3", "x es diferente de tres"),
    ("x²", "x al cuadrado"),
    ("x³", "x al cubo"),
    ("x⁻¹", "x a la menos uno"),
    ("½", "un medio"),
    ("2½", "dos y un medio"),
    ("x − 1", "x menos uno"),
    ("2**3", "dos al cubo"),
    ("sqrt(16)", "la raíz cuadrada de dieciséis"),
    # powers and roots
    ("x^4", "x a la cuarta"),
    ("10^6", "diez a la sexta"),
    ("x^12", "x a la doce"),
    ("x^-2", "x a la menos dos"),
    ("x^(n+1)", "x a la potencia n más uno"),
    ("e^(2)", "e al cuadrado"),
    ("x^y", "x a la ye"),
    ("√(x+1)", "la raíz cuadrada de la cantidad x más uno"),
    ("2√9", "dos por la raíz cuadrada de nueve"),
    ("∛8", "la raíz cúbica de ocho"),
    ("La raíz √16 es 4", "La raíz cuadrada de dieciséis es cuatro"),
    # grouping, functions, points
    ("(x+1)(x-1)", "la cantidad x más uno por la cantidad x menos uno"),
    ("(-3)(4)", "menos tres por cuatro"),
    ("(x+1)^2", "la cantidad x más uno, al cuadrado"),
    ("(x+1)/2", "la cantidad x más uno sobre dos"),
    ("x(x+1)", "x por la cantidad x más uno"),
    ("f(x) = 2x + 1", "f de x es igual a dos x más uno"),
    ("el punto (2, 3)", "el punto dos, tres"),
    ("(-1, 4)", "el punto menos uno, cuatro"),
    ("(x, y)", "el punto x, ye"),
    # algebra and the variable y
    ("x^2 + 2x + 1 = 0", "x al cuadrado más dos x más uno es igual a cero"),
    ("y = mx + b", "ye es igual a m x más b"),
    ("a² + 2ab + b²", "a al cuadrado más dos a b más b al cuadrado"),
    ("2y + 1", "dos ye más uno"),
    ("y - 3 = 0", "ye menos tres es igual a cero"),
    ("Despeja y: y = 2x", "Despeja ye: ye es igual a dos x"),
    ("x = 3 y y = 4", "x es igual a tres y ye es igual a cuatro"),
    ("Las soluciones son 3 y -2", "Las soluciones son tres y menos dos"),
    # other symbols
    ("x ± 2", "x más o menos dos"),
    ("x ≈ 3.14", "x es aproximadamente igual a tres punto catorce"),
    ("3x = 7 → x = 7/3", "tres x es igual a siete entonces x es igual a siete tercios"),
    ("-2/3", "menos dos tercios"),
    ("1/2x", "un medio x"),
    ("2πr", "dos pi r"),
    ("90°", "noventa grados"),
    ("1°", "un grado"),
    ("25 °C", "veinticinco grados Celsius"),
    ("12.5%", "doce punto cinco por ciento"),
    # numbers in prose
    ("1,250", "mil doscientos cincuenta"),
    ("1,000,000", "un millón"),
    ("3.14", "tres punto catorce"),
    ("0.05", "cero punto cero cinco"),
    ("0.125", "cero punto uno dos cinco"),
    ("21/2", "veintiún medios"),
    ("7/37", "siete sobre treinta y siete"),
    ("1 paso", "un paso"),
    ("1 vez", "una vez"),
    ("21 manzanas", "veintiuna manzanas"),
    ("200 manzanas", "doscientas manzanas"),
    ("1 problema", "un problema"),
    ("1er paso", "primer paso"),
    ("2do intento", "segundo intento"),
    ("3ra vez", "tercera vez"),
    # mixed prose
    ("Ahora resta 5 en ambos lados: 3x = 7", "Ahora resta cinco en ambos lados: tres x es igual a siete"),
    ("¿Cuánto es 2+2?", "¿Cuánto es dos más dos?"),
    ("**Paso 1:** resta 3", "Paso uno: resta tres"),
    ('"3"', '"tres"'),
    ("(por ejemplo, 3 manzanas)", "(por ejemplo, tres manzanas)"),
    ("- Primero suma 3", "- Primero suma tres"),
    ("y/o", "y o"),
    ("sí/no", "sí o no"),
    ("x-y", "x menos ye"),
]

PROSE = [
    "¿y ahora?",
    "No entiendo, ¿me ayudas?",
    "Es una auto-evaluación rápida.",
    "Piensa (sin prisa) en el siguiente paso.",
    "¡Excelente trabajo!",
    "",
]

NUMBERS = [
    (0, "cero"),
    (1, "uno"),
    (7, "siete"),
    (10, "diez"),
    (11, "once"),
    (15, "quince"),
    (16, "dieciséis"),
    (19, "diecinueve"),
    (20, "veinte"),
    (21, "veintiuno"),
    (22, "veintidós"),
    (23, "veintitrés"),
    (26, "veintiséis"),
    (29, "veintinueve"),
    (30, "treinta"),
    (31, "treinta y uno"),
    (45, "cuarenta y cinco"),
    (99, "noventa y nueve"),
    (100, "cien"),
    (101, "ciento uno"),
    (115, "ciento quince"),
    (200, "doscientos"),
    (342, "trescientos cuarenta y dos"),
    (500, "quinientos"),
    (777, "setecientos setenta y siete"),
    (999, "novecientos noventa y nueve"),
    (1000, "mil"),
    (1001, "mil uno"),
    (1250, "mil doscientos cincuenta"),
    (2026, "dos mil veintiséis"),
    (10000, "diez mil"),
    (21000, "veintiún mil"),
    (100000, "cien mil"),
    (101000, "ciento un mil"),
    (500500, "quinientos mil quinientos"),
    (999999, "novecientos noventa y nueve mil novecientos noventa y nueve"),
    (1000000, "un millón"),
    (2500000, "dos millones quinientos mil"),
    (-5, "menos cinco"),
]

FRACTIONS = [
    (1, 2, "un medio"),
    (3, 2, "tres medios"),
    (1, 3, "un tercio"),
    (2, 3, "dos tercios"),
    (1, 4, "un cuarto"),
    (3, 4, "tres cuartos"),
    (1, 5, "un quinto"),
    (2, 5, "dos quintos"),
    (1, 6, "un sexto"),
    (5, 6, "cinco sextos"),
    (1, 7, "un séptimo"),
    (3, 7, "tres séptimos"),
    (1, 8, "un octavo"),
    (5, 8, "cinco octavos"),
    (1, 9, "un noveno"),
    (4, 9, "cuatro novenos"),
    (1, 10, "un décimo"),
    (3, 10, "tres décimos"),
    (1, 11, "un onceavo"),
    (7, 11, "siete onceavos"),
    (5, 12, "cinco doceavos"),
    (1, 13, "un treceavo"),
    (2, 15, "dos quinceavos"),
    (3, 16, "tres dieciseisavos"),
    (1, 20, "un veinteavo"),
    (1, 21, "un veintiunavo"),
    (2, 23, "dos veintitresavos"),
    (5, 24, "cinco veinticuatroavos"),
    (1, 30, "un treintavo"),
    (3, 100, "tres centésimos"),
    (1, 1000, "un milésimo"),
    (21, 2, "veintiún medios"),
    (7, 37, "siete sobre treinta y siete"),
    (5, 1, "cinco sobre uno"),
    (3, 0, "tres sobre cero"),
]

CORPUS = [
    "¡Muy bien! Ahora resta 5 en ambos lados: 3x = 7.",
    "Divide ambos lados entre 3 para despejar x: x = 7/3.",
    "Paso 1: escribe la ecuación 2x + 4 = 10.",
    "Si x = 2, entonces 3x - 1 = 5.",
    "¿Qué número sumado a 1/2 da 3/4?",
    "La fracción 6/8 se simplifica a 3/4.",
    "Recuerda que 2(x + 3) = 2x + 6.",
    "El área de un círculo es A = πr².",
    "Usa la fórmula a² + b² = c² para el triángulo.",
    "La pendiente es m = (y2 - y1)/(x2 - x1).",
    "Marca el punto (2, 3) en la gráfica.",
    "La recta pasa por (-1, 4) y (0, 2).",
    "Si x < 5, el valor es menor; si x ≥ 5, es mayor o igual.",
    "Sabemos que x ≠ 0, así que podemos dividir.",
    "El 25% de 80 es 20.",
    "Multiplica 3 × 4 y luego suma 2.",
    "La raíz √16 es 4, porque 4 · 4 = 16.",
    "Tenemos 2 1/2 tazas de agua.",
    "Calcula 0.5 + 0.25 = 0.75.",
    "La temperatura del agua es 25 °C.",
    "¿Puedes resolver 5x - 3 = 2x + 9?",
    "Despeja y: y = -2x + 8.",
    "Factoriza x² - 9 = (x + 3)(x - 3).",
    "El exponente indica que 2^5 = 32.",
    "Si f(x) = x² + 1, entonces f(2) = 5.",
    "Hay 21 alumnos y 1 problema por resolver.",
    "**Paso 2:** divide 12 ÷ 4 = 3.",
    "Compara 3/5 y 4/7: ¿cuál es mayor?",
    "Redondea 3.14159 a 3.14.",
    "La suma de los ángulos es 180° en un triángulo.",
    "La solución es x = 4 ± 2.",
    "Tienes 1 intento más; luego 3x/4 = 6 → x = 8.",
    "x^2 - 5x + 6 = 0 tiene dos soluciones: x = 2 y x = 3.",
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_to_speech_es(text: str, expected: str) -> None:
    assert to_speech_es(text) == expected


def test_table_is_large_enough() -> None:
    assert len(CASES) >= 60


@pytest.mark.parametrize("text", [c[0] for c in CASES])
def test_idempotent_and_symbol_free(text: str) -> None:
    once = to_speech_es(text)
    assert to_speech_es(once) == once
    assert not FORBIDDEN.search(once), once


@pytest.mark.parametrize("text", PROSE)
def test_prose_untouched(text: str) -> None:
    assert to_speech_es(text) == text


@pytest.mark.parametrize(("n", "expected"), NUMBERS)
def test_number_words(n: int, expected: str) -> None:
    assert number_to_words_es(n) == expected


def test_number_words_agree_with_noun() -> None:
    assert number_to_words_es(1, apocope=True) == "un"
    assert number_to_words_es(21, apocope=True) == "veintiún"
    assert number_to_words_es(31, apocope=True) == "treinta y un"
    assert number_to_words_es(1, fem=True) == "una"
    assert number_to_words_es(31, fem=True) == "treinta y una"
    assert number_to_words_es(300, fem=True) == "trescientas"
    assert number_to_words_es(1_000_000, fem=True) == "un millón"


def test_huge_numbers_are_read_digit_by_digit() -> None:
    assert number_to_words_es(10**12) == "uno " + " ".join(["cero"] * 12)


@pytest.mark.parametrize(("num", "den", "expected"), FRACTIONS)
def test_fraction_words(num: int, den: int, expected: str) -> None:
    assert fraction_to_words_es(num, den) == expected


@pytest.mark.parametrize(("num", "den", "expected"), FRACTIONS)
def test_fraction_in_text(num: int, den: int, expected: str) -> None:
    assert to_speech_es(f"{num}/{den}") == expected


def test_corpus_is_spoken_cleanly() -> None:
    assert len(CORPUS) >= 30
    for sentence in CORPUS:
        spoken = to_speech_es(sentence)
        assert not FORBIDDEN.search(spoken), (sentence, spoken)
        assert to_speech_es(spoken) == spoken, sentence
        for word in re.findall(r"[^\W\d_]{4,}", sentence):
            assert word in spoken, (word, spoken)


def test_corpus_spot_checks() -> None:
    spoken = [to_speech_es(s) for s in CORPUS]
    assert spoken[1] == "Divide ambos lados entre tres para despejar x: x es igual a siete tercios."
    assert spoken[11] == "La recta pasa por el punto menos uno, cuatro y el punto cero, dos."
    assert spoken[25] == "Hay veintiún alumnos y un problema por resolver."
    assert spoken[26] == "Paso dos: divide doce entre cuatro es igual a tres."


@pytest.mark.parametrize("text", ["3x", "x^2", "x - y", "½", "2 + 2", "-4", "√9"])
def test_has_math_symbols(text: str) -> None:
    assert has_math_symbols(text)


@pytest.mark.parametrize("text", ["hola", "auto-evaluación", "¿y ahora?", "", "ayuda"])
def test_has_no_math_symbols(text: str) -> None:
    assert not has_math_symbols(text)


def test_private_use_input_is_not_trusted_as_markup() -> None:
    assert to_speech_es("x\ue000 = 2\ue00a") == "x es igual a dos"
