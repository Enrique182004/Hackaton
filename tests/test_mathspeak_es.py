from __future__ import annotations

import re
import sys
import time
import types

import pytest

from biobench_sar.llm import mathspeak_es
from biobench_sar.llm.mathspeak_es import fraction_to_words_es, has_math_symbols, number_to_words_es, to_speech_es

# No symbol may reach the voice, and no raw hyphen may sit between operands ("cinco - menos tres" sounds like 5-3).
# Word-internal hyphens ("auto-evaluación") and a leading "- " list bullet are prose and allowed. An en / em dash
# glued to a lone variable ("–b") or opening a line ("–cinco más tres") is a sign the voice would skip.
FORBIDDEN = re.compile(
    r"[\d^/*=+<>×·÷√∛%≤≥≠≈±²³½¼¾π|$#∈∉µμ\ue000-\uf8ff]|(?<![^\W\d_])-|-(?![^\W\d_])"
    r"|[–—‒](?=[a-zA-Z](?![^\W\d_]))|^[–—‒](?=\w)",
    re.M,
)
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
# Review round 2 (fix/mathspeak-es-review). Each group fails on the code before the fix.
NBSP, NNBSP = chr(0x00A0), chr(0x202F)
RANGES = [
    # An unspaced dash between two plain numbers is a range before a unit / noun, or with no other math around.
    ("Los trofozoítos miden 15–45 µm", "Los trofozoítos miden quince a cuarenta y cinco micrómetros"),
    ("range 25-37 °C", "range veinticinco a treinta y siete grados Celsius"),
    ("60-65 °C", "sesenta a sesenta y cinco grados Celsius"),
    ("100-500 mL", "cien a quinientos mililitros"),
    ("Toma 3–5 muestras", "Toma tres a cinco muestras"),
    ("cada 24–48 horas", "cada veinticuatro a cuarenta y ocho horas"),
    ("Tarda 2–3 días", "Tarda dos a tres días"),
    ("10–12 años", "diez a doce años"),
    ("Repite 2–3 veces.", "Repite dos a tres veces."),
    ("Dedica 5–10 minutos a practicar.", "Dedica cinco a diez minutos a practicar."),
    ("Para niños de 8–12 años.", "Para niños de ocho a doce años."),
    ("Del 3–7 de marzo.", "Del tres al siete de marzo."),
    ("del 5–10 %", "del cinco al diez por ciento"),
    ("entre 20–30 °C", "entre veinte y treinta grados Celsius"),
    ("pH 7–8", "pH siete a ocho"),
    ("Lee 2-3 libros", "Lee dos a tres libros"),
    ("1990-2000", "mil novecientos noventa a dos mil"),
    (
        "Incuba a 25-37 °C durante 24-48 h.",
        "Incuba a veinticinco a treinta y siete grados Celsius durante veinticuatro a cuarenta y ocho horas.",
    ),
    ("Toma 1–2 tazas", "Toma una a dos tazas"),
    ("21–25 alumnas", "veintiuna a veinticinco alumnas"),
    ("1.5-2.5 mL", "uno punto cinco a dos punto cinco mililitros"),
    ("1-2 h", "una a dos horas"),
    # ...and a subtraction next to a relation, another operator, a math cue or a group.
    ("8–3 = 5", "ocho menos tres es igual a cinco"),
    ("x = 8-3", "x es igual a ocho menos tres"),
    ("2x-3=7", "dos x menos tres es igual a siete"),
    ("2-3 + 1", "dos menos tres más uno"),
    ("Calcula 8-3", "Calcula ocho menos tres"),
    ("¿Cuánto es 8-3?", "¿Cuánto es ocho menos tres?"),
    ("8-3 es 5", "ocho menos tres es cinco"),
    ("8-3", "ocho menos tres"),
    ("2(8-3)", "dos por la cantidad ocho menos tres"),
    ("(8-3)^2", "la cantidad ocho menos tres, al cuadrado"),
    ("√(1-2)", "la raíz cuadrada de la cantidad uno menos dos"),
    ("3-5x", "tres menos cinco x"),
    ("3-5 x", "tres menos cinco x"),
    ("Si x = 2, entonces 3-5", "Si x es igual a dos, entonces tres menos cinco"),
    ("Toma 3-5 muestras y calcula 8-3 = 5", "Toma tres a cinco muestras y calcula ocho menos tres es igual a cinco"),
    ("x - 4", "x menos cuatro"),
    ("9 – 4", "nueve menos cuatro"),
    ("5-3", "cinco menos tres"),
    ("3-4", "tres a cuatro"),
    # A measurement after a relation is still a range; a one-letter unit there may be a variable, so it subtracts.
    ("Dura 2-3 días = 48-72 horas", "Dura dos a tres días es igual a cuarenta y ocho a setenta y dos horas"),
    ("La distancia ≈ 2-3 km", "La distancia es aproximadamente igual a dos a tres kilómetros"),
    ("x = 3-5 m", "x es igual a tres menos cinco m"),
    ("t = 72-48 horas", "t es igual a setenta y dos menos cuarenta y ocho horas"),
    # The end of a range carries a one-letter unit.
    ("5-10 g", "cinco a diez gramos"),
    ("Mide 2–3 m", "Mide dos a tres metros"),
    ("de 2 a 3 m", "de dos a tres metros"),
    ("entre 2 y 3 m", "entre dos y tres metros"),
    ("de 2 a 3 m + 1", "de dos a tres m más uno"),
]
ASIDES = [
    # A spaced em / en dash before a Spanish one-letter word is an aside, not a minus (and "y" stays "i").
    (
        "¡Correcto! 3 + 4 = 7 — y lo hiciste rapidísimo.",
        "¡Correcto! tres más cuatro es igual a siete — y lo hiciste rapidísimo.",
    ),
    ("x = 5 — y ahora comprueba.", "x es igual a cinco — y ahora comprueba."),
    ("Obtuviste 10 — a ver el siguiente.", "Obtuviste diez — a ver el siguiente."),
    ("Paso 2 — a la derecha.", "Paso dos — a la derecha."),
    ("Son 3 – y no 4.", "Son tres – y no cuatro."),
    ("En 1990 — y hasta 2000 — fue así.", "En mil novecientos noventa — y hasta dos mil — fue así."),
    ("Paso 3 — b) el siguiente", "Paso tres — b) el siguiente"),
    # A lone variable that ends the expression is still subtracted.
    ("x = 5 — y", "x es igual a cinco menos ye"),
    ("(x – a)^2", "la cantidad x menos a, al cuadrado"),
]
SIGNED_DASHES = [
    ("x = –b", "x es igual a menos b"),
    ("–5 + 3 = –2", "menos cinco más tres es igual a menos dos"),
    (
        "x = –b ± √(b² – 4ac) / 2a",
        "x es igual a menos b más o menos la raíz cuadrada de la cantidad b al cuadrado menos cuatro a c sobre dos a",
    ),
    # An en dash glued to a number is its sign before a degree, a percent, the end of the clause or another number.
    ("–5", "menos cinco"),
    ("–5 °C", "menos cinco grados Celsius"),
    ("Hace –5 grados", "Hace menos cinco grados"),
    ("–5, –3 y 0", "menos cinco, menos tres y cero"),
    ("de –5 a 10 °C", "de menos cinco a diez grados Celsius"),
]
MONEY_MAGNITUDES = [
    ("$10 mil", "diez mil dólares"),
    ("Ganó $1 millón", "Ganó un millón de dólares"),
    ("$3 millones", "tres millones de dólares"),
    ("Un terreno de $1.5 millones", "Un terreno de uno punto cinco millones de dólares"),
    ("Cuesta $5.5 mil", "Cuesta cinco punto cinco mil dólares"),
    ("Cuesta $2 mil pesos", "Cuesta dos mil pesos"),
    ("$3 millones de pesos", "tres millones de pesos"),
    ("$21 mil", "veintiún mil dólares"),
    ("$3 mil millones", "tres mil millones de dólares"),
    ("$1 mil", "mil dólares"),
    ("$-5", "menos cinco dólares"),
    ("La respuesta es $2x$", "La respuesta es dos x"),
    ("La respuesta es $x$.", "La respuesta es x."),
    ("Cuesta $5 y $10", "Cuesta cinco dólares y diez dólares"),
    ("Ganó $1 millón y medio", "Ganó un millón y medio de dólares"),
    ("$2 millones y medio de pesos", "dos millones y medio de pesos"),
    ("$2x + 1", "dos x más uno"),
    ("La respuesta es $2x", "La respuesta es dos x"),
]
THOUSANDS = [
    ("Hay 25 000 habitantes", "Hay veinticinco mil habitantes"),
    (f"Hay 25{NBSP}000 habitantes", "Hay veinticinco mil habitantes"),
    (f"Hay 25{NNBSP}000 habitantes", "Hay veinticinco mil habitantes"),
    ("25 000", "veinticinco mil"),
    ("1 250 000", "un millón doscientos cincuenta mil"),
    ("Cuesta $10 000", "Cuesta diez mil dólares"),
    ("El auto cuesta $250 000 pesos", "El auto cuesta doscientos cincuenta mil pesos"),
    ("Escala 1 : 100 000", "Escala uno es a cien mil"),
    # Groups that are not exactly three digits, or a list of numbers, stay separate.
    ("3 4", "tres cuatro"),
    ("Los números 3 4 5", "Los números tres cuatro cinco"),
    ("Los números 10 100 1000", "Los números diez cien mil"),
    ("Los números 100 200 300", "Los números cien doscientos trescientos"),
    ("Los números 5 10 100", "Los números cinco diez cien"),
]
Y_IN_MATH = [
    ("Calcula y luego suma 3.", "Calcula y luego suma tres."),
    ("Calcula y después divide entre 2.", "Calcula y después divide entre dos."),
    ("Vale y ahora suma 2.", "Vale y ahora suma dos."),
    ("Resta 3 y divide", "Resta tres y divide"),
    ("el valor de y", "el valor de ye"),
    ("despeja y", "despeja ye"),
    ("Resuelve para y: 2y = 4", "Resuelve para ye: dos ye es igual a cuatro"),
    ("Halla y cuando x = 3", "Halla ye cuando x es igual a tres"),
    ("Calcula y si x = 2", "Calcula ye si x es igual a dos"),
    ("Encuentra y de la ecuación 2y = 6", "Encuentra ye de la ecuación dos ye es igual a seis"),
    ("y ∈ [0, 1]", "ye pertenece al intervalo de cero a uno"),
    ("Encuentra y sabiendo que x = 2", "Encuentra ye sabiendo que x es igual a dos"),
    # Right after the variable y, another y is the conjunction; after the conjunction it is the variable.
    ("el valor de y y el de x", "el valor de ye y el de x"),
    ("despeja y y luego comprueba", "despeja ye y luego comprueba"),
    ("x y y son enteros", "x y ye son enteros"),
    ("Calcula x y y", "Calcula x y ye"),
    ("Despeja y en 3y = 9", "Despeja ye en tres ye es igual a nueve"),
    ("Despeja y, luego calcula x.", "Despeja ye, luego calcula x."),
]
GENDER = [
    ("Tienes 1 error en el paso 2", "Tienes un error en el paso dos"),
    ("Hay 1 error y 21 aciertos", "Hay un error y veintiún aciertos"),
    ("Te doy 1 pista", "Te doy una pista"),
    ("Usa 1 calculadora", "Usa una calculadora"),
    ("Hay 101 dálmatas", "Hay ciento un dálmatas"),
    ("Hay 1 tortuga y 1 hexágono", "Hay una tortuga y un hexágono"),
    ("Dibuja 1 cuadro", "Dibuja un cuadro"),
    ("Toma 1 muestra de agua", "Toma una muestra de agua"),
    # A "noun" that is really a verb or the adverb "solo" never agrees.
    ("Al 21 suma 4.", "Al veintiuno suma cuatro."),
    ("A 200 resta 50.", "A doscientos resta cincuenta."),
    ("De 500 resta 200.", "De quinientos resta doscientos."),
    ("x vale 1 solo cuando x es positivo", "x vale uno solo cuando x es positivo"),
    ("El 1 solo se divide entre sí mismo", "El uno solo se divide entre sí mismo"),
    ("La figura 1 muestra que x = 2", "La figura uno muestra que x es igual a dos"),
    ("1 solo paso", "un solo paso"),
    ("Toma 1 tableta", "Toma una tableta"),
    ("Toma 2-3 veces al día 1 pastilla", "Toma dos a tres veces al día una pastilla"),
    ("Usa 1 frasco", "Usa un frasco"),
]
TIMES_AND_RATIOS = [
    ("Escala 1:50,000", "Escala uno es a cincuenta mil"),
    ("La película dura 2:30.", "La película dura dos y treinta."),
    ("El examen es 10:30", "El examen es diez y treinta"),
    ("La clase empieza 8:00", "La clase empieza ocho en punto"),
    ("Nos vemos a la 1", "Nos vemos a la una"),
    ("Es la 1.", "Es la una."),
    ("Llega a la 1 de la tarde", "Llega a la una de la tarde"),
    ("Diluye 1:10", "Diluye uno es a diez"),
    ("La razón es 3:45", "La razón es tres es a cuarenta y cinco"),
    ("Simplifica 2:30", "Simplifica dos es a treinta"),
    ("La clase termina 9:50", "La clase termina nueve y cincuenta"),
    ("x elevado a la 1", "x elevado a la uno"),
]
MEASUREMENTS = [
    (
        "Laguna de Bustillos tiene 1.5 m de profundidad",
        "Laguna de Bustillos tiene uno punto cinco metros de profundidad",
    ),
    ("Un rectángulo de 3 m por 5 m", "Un rectángulo de tres metros por cinco metros"),
    ("Un rectángulo de 3 m x 5 m", "Un rectángulo de tres metros por cinco metros"),
    ("Compra 1 L de leche", "Compra un litro de leche"),
    ("15 µm", "quince micrómetros"),
    ("3 μm", "tres micrómetros"),
    ("5 µL", "cinco microlitros"),
    ("Incuba 1 h", "Incuba una hora"),
    ("30 min", "treinta minutos"),
    ("2.3 × 10^4 células/mL", "dos punto tres por diez a la cuarta células por mililitro"),
    ("10 metros/segundo", "diez metros por segundo"),
    ("3 pesos/kilo", "tres pesos por kilo"),
    ("50 litros/minuto", "cincuenta litros por minuto"),
    ("Cuesta $15/kg", "Cuesta quince dólares por kilo"),
    ("2 g/cm³", "dos gramos por centímetro cúbico"),
    ("5 mg/L", "cinco miligramos por litro"),
    ("x = m/g", "x es igual a m sobre g"),
    ("alumnas/alumno", "alumnas o alumno"),
]
SETS_AND_INTERVALS = [
    ("[2, 5]", "el intervalo de dos a cinco"),
    ("(2, 5]", "el intervalo de dos a cinco"),
    ("x ∈ [0, 1]", "x pertenece al intervalo de cero a uno"),
    ("El conjunto {1, 2, 3}", "El conjunto uno, dos, tres"),
    ("A = {1, 2, 3}", "A es igual al conjunto uno, dos, tres"),
    ("{[2 + 3] × 4}", "la cantidad, la cantidad dos más tres por cuatro"),
    ("Los puntos (1, 2) y (3, 4)", "Los puntos uno, dos y tres, cuatro"),
    ("{1, 2, 3, ...}", "el conjunto uno, dos, tres, etcétera"),
]
OLDER = [
    ("Es el 1° de mayo.", "Es el primero de mayo."),
    ("El 1° de septiembre", "El primero de septiembre"),
    ("El 2° de mayo", "El dos de mayo"),
    ("El COVID-19", "El COVID diecinueve"),
    ("xy = 6", "x ye es igual a seis"),
    ("3 x 4 = 12", "tres por cuatro es igual a doce"),
    ("Hay 3x10^5 amibas", "Hay tres por diez a la quinta amibas"),
    ("Tu calificación es 10/10", "Tu calificación es diez de diez"),
    ("Contestaste 7/10 bien", "Contestaste siete de diez bien"),
    ("Tienes 8/10 correctas", "Tienes ocho de diez correctas"),
    ("ab-1", "ab menos uno"),
    ("Recuerda: el 1° de cada mes revisamos la tarea.", "Recuerda: el primero de cada mes revisamos la tarea."),
    ("El 1° del mes", "El primero del mes"),
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
    + RANGES
    + ASIDES
    + SIGNED_DASHES
    + MONEY_MAGNITUDES
    + THOUSANDS
    + Y_IN_MATH
    + GENDER
    + TIMES_AND_RATIOS
    + MEASUREMENTS
    + SETS_AND_INTERVALS
    + OLDER
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
    # The conjunction "y" after a tutor's verb stays "i": only a math cue makes it the variable.
    "Encuentra y dime tu resultado.",
    "Calcula y responde la pregunta.",
    "Calcula y factoriza la expresión.",
    "Encuentra y reduce términos semejantes.",
    "Para y piensa antes de contestar.",
    "Bueno, y.",
    "Primero calcula y luego suma.",
    "Lee con atención, calcula y responde.",
    "Calcula y contesta.",
    "Encuentra y encierra la respuesta.",
    "Encuentra y después escribe el resultado.",
    "¿Cuánto vale y cuánto cuesta?",
    "Vale y ahora piensa.",
    "Encuentra x y comprueba",
    "Calcula el valor y anótalo.",
    "Calcula y de ahí despeja x.",
    "Encuentra y para comprobar.",
    # Seen live from qwen2.5vl:3b: a second instruction right after "despeja y" makes it the conjunction.
    "Despeja y calcula la cantidad de carbono en el suelo.",
    "Hay que despejar y comprobar el resultado.",
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


def test_many_ranges_stay_linear() -> None:
    # Each range used to rescan its whole line for a relation: 33 KB of "1990–2000, " took about 2 s.
    text = "1990–2000, " * 3000
    start = time.perf_counter()
    spoken = to_speech_es(text)
    assert time.perf_counter() - start < 0.5
    assert spoken.startswith("mil novecientos noventa a dos mil, ")


def test_membership_is_math() -> None:
    assert has_math_symbols("x ∈ A")
    assert to_speech_es("x ∉ A") == "x no pertenece a A"


def test_amount_glued_to_a_variable_is_not_money() -> None:
    assert to_speech_es("$2x") == "dos x"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("—5 minutos más, dijo", "—cinco minutos más, dijo"),
        ("Es –5 veces– mayor", "Es –cinco veces– mayor"),
        ("Paso 2 –5 minutos", "Paso dos –cinco minutos"),
    ],
)
def test_dialogue_and_aside_dashes_before_a_number_stay_prose(text: str, expected: str) -> None:
    # An em dash, a dash before a noun or a dash after another number is not a sign.
    assert to_speech_es(text) == expected


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
