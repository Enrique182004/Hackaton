from __future__ import annotations

import re
import sys
import time
import types

import pytest

from biobench_sar.llm import mathspeak_es
from biobench_sar.llm.mathspeak_es import fraction_to_words_es, has_math_symbols, number_to_words_es, to_speech_es

# No symbol may reach the voice, and no raw hyphen may sit between operands ("cinco - menos tres" sounds like 5-3).
# Word-internal hyphens ("auto-evaluación") and a leading "- " list bullet are prose and allowed.
FORBIDDEN = re.compile(r"[\d^/*=+<>×·÷√∛%≤≥≠≈±²³½¼¾π|$#\ue000-\uf8ff]|(?<![^\W\d_])-|-(?![^\W\d_])")
_BULLET = re.compile(r"(?m)^[ \t]*- ")


def _symbol(spoken: str) -> re.Match[str] | None:
    return FORBIDDEN.search(_BULLET.sub("", spoken))


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

# Review regressions (feat/mathspeak-es). Each group fails on the code before the fix.
POWERED_GROUPS = [
    # (-3)² and -3² are the distinction a teacher stresses: the base is the whole group, never "por al cuadrado".
    ("(-3)^2", "la cantidad menos tres, al cuadrado"),
    ("(-3)² = 9", "la cantidad menos tres, al cuadrado es igual a nueve"),
    ("-3^2", "menos tres al cuadrado"),
    ("(3x)² = 9x²", "la cantidad tres x, al cuadrado es igual a nueve x al cuadrado"),
    ("(1/2)²", "la cantidad un medio, al cuadrado"),
    ("(-1)^n", "la cantidad menos uno, a la n"),
    ("(x)^2", "la cantidad x, al cuadrado"),
    ("(2a)^2", "la cantidad dos a, al cuadrado"),
    ("(ab)^2", "la cantidad a b, al cuadrado"),
    ("((x+1))^2", "la cantidad x más uno, al cuadrado"),
    ("|x|^2", "el valor absoluto de x, al cuadrado"),
]
NEGATIVE_OPERANDS = [
    ("2 * -3", "dos por menos tres"),
    ("2*-3", "dos por menos tres"),
    ("-2*-3", "menos dos por menos tres"),
    ("2 · -3", "dos por menos tres"),
    ("x * -1", "x por menos uno"),
    ("5 - -3", "cinco menos menos tres"),
    ("x - - 3", "x menos menos tres"),
    ("x = - 4", "x es igual a menos cuatro"),
    ("2 ** 3", "dos al cubo"),
    ("x ** 2", "x al cuadrado"),
    ("**3** y **4**", "tres y cuatro"),
]
DASHES = [
    ("8 – 3 = 5", "ocho menos tres es igual a cinco"),
    ("x – 1", "x menos uno"),
    ("x — 1", "x menos uno"),
    ("x ‒ 1", "x menos uno"),
    ("x – 1 = 0", "x menos uno es igual a cero"),
    ("x² – 1", "x al cuadrado menos uno"),
    ("x = –3", "x es igual a menos tres"),
    ("Calcula 8–3", "Calcula ocho menos tres"),
    ("1990–2000", "mil novecientos noventa a dos mil"),
    ("Hay 3 opciones —1, 2 o 3— para elegir", "Hay tres opciones —uno, dos o tres— para elegir"),
]
INCLUSIVE = [
    ("¿Estás listo/a?", "¿Estás listo o lista?"),
    ("¿Estás atorado/a? Te ayudo.", "¿Estás atorado o atorada? Te ayudo."),
    ("Bienvenido/a", "Bienvenido o Bienvenida"),
    ("Eres muy bueno/a", "Eres muy bueno o buena"),
    ("¡Eres un campeón/a!", "¡Eres un campeón o campeona!"),
    ("Pregúntale a tu maestro/a", "Pregúntale a tu maestro o maestra"),
    ("alumno/a", "alumno o alumna"),
    ("los/las", "los o las"),
    ("los/las alumnos/as", "los o las alumnos o alumnas"),
    ("¿Estás seguro(a)?", "¿Estás seguro o segura?"),
    ("Los niño(a)s ganaron 3 puntos", "Los niños o niñas ganaron tres puntos"),
    ("Revisa tu(s) respuesta(s).", "Revisa tu respuesta."),
    ("x/a", "x sobre a"),
    ("cos(a)", "coseno de a"),
]
AGREEMENT = [
    # Plain contract form unless a known noun follows: never "una da", "un entonces", "doscientas representa".
    ("3 + 1 da 4", "tres más uno da cuatro"),
    ("Si restas 1 queda 4", "Si restas uno queda cuatro"),
    ("El 1 representa el total", "El uno representa el total"),
    ("Si 200 representa el total", "Si doscientos representa el total"),
    ("x = 1 entonces", "x es igual a uno entonces"),
    ("Divide entre 1 ambos lados", "Divide entre uno ambos lados"),
    ("Pon 1 arriba", "Pon uno arriba"),
    ("Suma 1 otra vez", "Suma uno otra vez"),
    ("x = 1 nunca", "x es igual a uno nunca"),
    ("200 + 300 da 500", "doscientos más trescientos da quinientos"),
    ("El 21 está bien", "El veintiuno está bien"),
    ("Multiplica por 1 cada lado", "Multiplica por uno cada lado"),
    ("x = 1 significa", "x es igual a uno significa"),
    ("Haz 1 resta", "Haz una resta"),
    ("1 sola vez", "una sola vez"),
    ("21 alumnas", "veintiuna alumnas"),
    ("1 ecuación", "una ecuación"),
    ("1000000 personas", "un millón de personas"),
]
CURRENCY = [
    ("Juan tiene $50 y gasta $15.", "Juan tiene cincuenta dólares y gasta quince dólares."),
    ("Cuesta $3.50", "Cuesta tres dólares con cincuenta centavos"),
    ("$1", "un dólar"),
    ("$21", "veintiún dólares"),
    ("$0.75", "setenta y cinco centavos"),
    ("$1.01", "un dólar con un centavo"),
    ("$1,000,000", "un millón de dólares"),
    ("$5 - $3 = $2", "cinco dólares menos tres dólares es igual a dos dólares"),
    ("$50 pesos", "cincuenta pesos"),
    ("$50 MXN", "cincuenta pesos"),
    ("Las manzanas cuestan $5 c/u", "Las manzanas cuestan cinco dólares cada uno"),
]
GROUPS_AND_POINTS = [
    ("la solución (x = 3)", "la solución x es igual a tres"),
    ("Sustituye (x = 2) en la ecuación", "Sustituye x es igual a dos en la ecuación"),
    ("(x > 2)", "x es mayor que dos"),
    ("el intervalo (2, 5)", "el intervalo de dos a cinco"),
    ("el intervalo [2, 5]", "el intervalo de dos a cinco"),
    ("el intervalo (-∞, 3)", "el intervalo de menos infinito a tres"),
    ("El punto A(2, 3)", "El punto A de coordenadas dos, tres"),
    ("P(1, -2)", "P de coordenadas uno, menos dos"),
    ("x = (2, 3)", "x es igual al punto dos, tres"),
    ("De (0, 0) a (3, 4)", "Del punto cero, cero al punto tres, cuatro"),
    ("f(2, 3)", "f de dos, tres"),
]
VARIABLE_Y = [
    ("despeja la y", "despeja la ye"),
    ("el valor de la y", "el valor de la ye"),
    ("¿cuánto vale y?", "¿cuánto vale ye?"),
    ("Encuentra x y y.", "Encuentra x y ye."),
    ("Los ejes son x, y", "Los ejes son x, ye"),
    ("Y = 2x", "ye es igual a dos x"),
    ("Encuentra y en 2x + y = 5", "Encuentra ye en dos x más ye es igual a cinco"),
    ("√y", "la raíz cuadrada de ye"),
    ("2√y", "dos por la raíz cuadrada de ye"),
]
UNITS_AND_FUNCTIONS = [
    ("El área es 12 cm²", "El área es doce centímetros cuadrados"),
    ("12 m²", "doce metros cuadrados"),
    ("1 m²", "un metro cuadrado"),
    ("8 cm³", "ocho centímetros cúbicos"),
    ("60 km/h", "sesenta kilómetros por hora"),
    ("9.8 m/s²", "nueve punto ocho metros por segundo al cuadrado"),
    ("21 km", "veintiún kilómetros"),
    ("3m + 2", "tres m más dos"),
    ("sen(x)", "seno de x"),
    ("3sen(x)", "tres seno de x"),
    ("sen x", "seno de x"),
    ("log(100)", "logaritmo de cien"),
    ("ln(e) = 1", "logaritmo natural de e es igual a uno"),
    ("cos(30°)", "coseno de treinta grados"),
    ("tan(45°) = 1", "tangente de cuarenta y cinco grados es igual a uno"),
    ("sen²(x) + cos²(x) = 1", "seno cuadrado de x más coseno cuadrado de x es igual a uno"),
    ("2pi", "dos pi"),
]
ORDINAL_DEGREES = [
    ("3° de secundaria", "tercero de secundaria"),
    ("1° grado", "primer grado"),
    ("2° lugar", "segundo lugar"),
    ("Estoy en 1° de prepa", "Estoy en primero de prepa"),
]
HALVES = [
    ("1/2 taza", "media taza"),
    ("1/2 hora", "media hora"),
    ("1/2 cucharada", "media cucharada"),
    ("1/2 pizza", "media pizza"),
    ("1/2 kilo", "medio kilo"),
    ("2 1/2 tazas", "dos y media tazas"),
    ("3/4 taza", "tres cuartos de taza"),
    ("2/3 partes", "dos terceras partes"),
]
NOTATION = [
    ("|x| = 3", "el valor absoluto de x es igual a tres"),
    ("|x - 2| = 5", "el valor absoluto de la cantidad x menos dos es igual a cinco"),
    ("2|x|", "dos por el valor absoluto de x"),
    ("2[x + (3 - 1)]", "dos por la cantidad x más la cantidad tres menos uno"),
    ("{x + 1}", "la cantidad x más uno"),
    ("2:3", "dos es a tres"),
    ("4 : 2", "cuatro es a dos"),
    ("2:3 = 4:6", "dos es a tres como cuatro es a seis"),
    ("8 : 2 = 4", "ocho entre dos es igual a cuatro"),
    ("a las 3:30", "a las tres y treinta"),
    ("a la 1:15", "a la una y quince"),
    ("Paso 1: 3x = 6", "Paso uno: tres x es igual a seis"),
    ("Ejercicio #2", "Ejercicio número dos"),
    ("páginas 1-5", "páginas uno a cinco"),
    ("Haz los ejercicios 3–7", "Haz los ejercicios tres a siete"),
]
MARKDOWN = [
    ("**ojo**: 3x = 6", "ojo: tres x es igual a seis"),
    ("**mal**", "mal"),
    ("*Nota:* 3x = 6", "Nota: tres x es igual a seis"),
]
REVIEW_CASES = (
    POWERED_GROUPS
    + NEGATIVE_OPERANDS
    + DASHES
    + INCLUSIVE
    + AGREEMENT
    + CURRENCY
    + GROUPS_AND_POINTS
    + VARIABLE_Y
    + UNITS_AND_FUNCTIONS
    + ORDINAL_DEGREES
    + HALVES
    + NOTATION
    + MARKDOWN
)

PROSE = [
    "¿y ahora?",
    "No entiendo, ¿me ayudas?",
    "Es una auto-evaluación rápida.",
    "Piensa (sin prisa) en el siguiente paso.",
    "¡Excelente trabajo!",
    "",
    "Bien – sigue así.",
    "Juan y María fueron al mercado.",
    "Encuentra y anota el resultado.",
    "¿Y?",
    "Y luego, ¿qué sigue?",
    "Lo hiciste sin(querer)",
    "| Paso | Resultado |",
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


@pytest.mark.parametrize(("text", "expected"), REVIEW_CASES)
def test_review_regressions(text: str, expected: str) -> None:
    assert to_speech_es(text) == expected


@pytest.mark.parametrize("text", [c[0] for c in CASES + REVIEW_CASES])
def test_idempotent_and_symbol_free(text: str) -> None:
    once = to_speech_es(text)
    assert to_speech_es(once) == once
    assert not _symbol(once), once


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
        assert not _symbol(spoken), (sentence, spoken)
        assert to_speech_es(spoken) == spoken, sentence
        for word in re.findall(r"[^\W\d_]{4,}", sentence):
            assert word in spoken, (word, spoken)


def test_corpus_spot_checks() -> None:
    spoken = [to_speech_es(s) for s in CORPUS]
    assert spoken[1] == "Divide ambos lados entre tres para despejar x: x es igual a siete tercios."
    assert spoken[11] == "La recta pasa por el punto menos uno, cuatro y el punto cero, dos."
    assert spoken[25] == "Hay veintiún alumnos y un problema por resolver."
    assert spoken[26] == "Paso dos: divide doce entre cuatro es igual a tres."


@pytest.mark.parametrize("text", ["3x", "x^2", "x - y", "½", "2 + 2", "-4", "√9", "|x|"])
def test_has_math_symbols(text: str) -> None:
    assert has_math_symbols(text)


@pytest.mark.parametrize("text", ["hola", "auto-evaluación", "¿y ahora?", "", "ayuda"])
def test_has_no_math_symbols(text: str) -> None:
    assert not has_math_symbols(text)


def test_private_use_input_is_not_trusted_as_markup() -> None:
    assert to_speech_es("x\ue000 = 2\ue00a") == "x es igual a dos"


def test_negative_base_is_not_the_negated_power() -> None:
    assert to_speech_es("(-3)^2") != to_speech_es("-3^2")
    assert "por al" not in to_speech_es("(-3)^2 = 9 y (3x)^2 = 9x^2")


def test_currency_word_is_configurable() -> None:
    assert to_speech_es("$50", currency="peso") == "cincuenta pesos"
    assert to_speech_es("$1.50", currency="peso") == "un peso con cincuenta centavos"
    with pytest.raises(ValueError):
        to_speech_es("$5", currency="euro")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1" * 5000, " ".join(["uno"] * 5000)),
        ("x^" + "9" * 5000, "x a la " + " ".join(["nueve"] * 5000)),
        ("1/" + "9" * 5000, "uno sobre " + " ".join(["nueve"] * 5000)),
    ],
    ids=["digits", "exponent", "denominator"],
)
def test_huge_digit_runs_are_read_digit_by_digit(text: str, expected: str) -> None:
    assert to_speech_es(text) == expected


def test_internal_failure_falls_back_to_the_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(text: str, currency: tuple[str, str]) -> str:
        raise RuntimeError(text)

    monkeypatch.setattr(mathspeak_es, "_speak", boom)
    assert to_speech_es("2 + 2") == "2 + 2"


def test_deep_nesting_stays_fast() -> None:
    text = "(" * 1000 + "x" + ")" * 1000
    start = time.perf_counter()
    to_speech_es(text)
    assert time.perf_counter() - start < 0.5


def test_has_math_symbols_defers_to_the_shared_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = types.ModuleType("biobench_sar.llm.mathspeak")
    shared.has_math_symbols = lambda text: "∑" in text  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.llm.mathspeak", shared)
    mathspeak_es._shared_detector.cache_clear()
    try:
        assert has_math_symbols("∑ k")
        assert has_math_symbols("3x")
        assert not has_math_symbols("hola")
    finally:
        mathspeak_es._shared_detector.cache_clear()
