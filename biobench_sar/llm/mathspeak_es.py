"""Spanish (es-MX) phonetic math for TTS: "3x+5=12" -> "tres x más cinco es igual a doce"."""

from __future__ import annotations

import functools
import importlib
import re
from collections.abc import Callable

__all__ = ["fraction_to_words_es", "has_math_symbols", "number_to_words_es", "to_speech_es"]

# Private-use sentinels carry structure between passes; every one becomes words or a soft space at the end.
LQ, RQ = "\ue000", "\ue001"  # (x+1)  -> "la cantidad x más uno"
LS, RS = "\ue002", "\ue003"  # (3)    -> single term, parens dropped
LF, RF = "\ue004", "\ue005"  # f(x)   -> "f de x"
LE, RE = "\ue006", "\ue007"  # x^(n+1) -> compound exponent
OS, OE = "\ue008", "\ue009"  # an operand already spelled out (still an operand for minus / implicit-× decisions)
TIMES = "\ue00a"
SOFT = "\ue00b"  # a space we introduced: collapsible, dropped next to punctuation
FN = "\ue00c"  # sits between "sen" and its "(" so the group reads "seno de x"
ABS = "\ue00d"  # |x| -> "el valor absoluto de x"
YV = "\ue00e"  # the variable y, read "ye" (a bare "y" is the conjunction "i" to a Spanish voice)
EL = "\ue00f"  # the article of "el punto" / "el intervalo" / "el conjunto"; contracts to "al" / "del"
# Prose brackets parked while grouping, one sentinel per bracket so each comes back as written.
_PARK = {"(": "\ue010", ")": "\ue011", "[": "\ue012", "]": "\ue013", "{": "\ue014", "}": "\ue015", "|": "\ue016"}
_UNPARK = str.maketrans({v: k for k, v in _PARK.items()})
_CLOSERS = RQ + RS + RF + RE + OE
_OPENERS = LQ + LS + LF + LE
_PREFIX = "√∛∜" + ABS
_OPS = "=+-*/×·÷<>≤≥≠≈±^"
_RELATIONS = "=<>≤≥≠≈"
_MAX_DEPTH = 24  # deeper nesting stays prose: each level is a full pass over the text
_MAX_DIGITS = 12  # longer digit runs are read digit by digit, never through int()

# fmt: off
_UNITS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
    "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno",
    "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
    "veintinueve",
]
# fmt: on
_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_HUNDREDS = {
    1: "ciento",
    2: "doscientos",
    3: "trescientos",
    4: "cuatrocientos",
    5: "quinientos",
    6: "seiscientos",
    7: "setecientos",
    8: "ochocientos",
    9: "novecientos",
}
_DENOMS = {
    2: "medio",
    3: "tercio",
    4: "cuarto",
    5: "quinto",
    6: "sexto",
    7: "séptimo",
    8: "octavo",
    9: "noveno",
    10: "décimo",
    21: "veintiunavo",
    100: "centésimo",
    1000: "milésimo",
}
_POWER_ORDINALS = {
    0: "cero",
    1: "primera",
    4: "cuarta",
    5: "quinta",
    6: "sexta",
    7: "séptima",
    8: "octava",
    9: "novena",
    10: "décima",
}
# n -> (masculine, feminine, apocope before a noun, accepted written suffixes)
_ORDINALS = {
    1: ("primero", "primera", "primer", ("er", "ero", "era", "ro", "ra")),
    2: ("segundo", "segunda", "segundo", ("do", "da")),
    3: ("tercero", "tercera", "tercer", ("er", "ero", "era", "ro", "ra")),
    4: ("cuarto", "cuarta", "cuarto", ("to", "ta")),
    5: ("quinto", "quinta", "quinto", ("to", "ta")),
    6: ("sexto", "sexta", "sexto", ("to", "ta")),
    7: ("séptimo", "séptima", "séptimo", ("mo", "ma")),
    8: ("octavo", "octava", "octavo", ("vo", "va")),
    9: ("noveno", "novena", "noveno", ("no", "na")),
    10: ("décimo", "décima", "décimo", ("mo", "ma")),
}
_UNICODE_FRACTIONS = {
    "½": "1/2",
    "⅓": "1/3",
    "⅔": "2/3",
    "¼": "1/4",
    "¾": "3/4",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻ⁿ", "0123456789-n")
_UNACCENT = str.maketrans("áéíóú", "aeiou")
_OPERATOR_WORDS = {
    "+": "más",
    "=": "es igual a",
    "<": "es menor que",
    ">": "es mayor que",
    "≤": "es menor o igual que",
    "≥": "es mayor o igual que",
    "≠": "es diferente de",
    "≈": "es aproximadamente igual a",
    "±": "más o menos",
    "×": "por",
    "÷": "entre",
    "->": "entonces",
    "=>": "entonces",
    "→": "entonces",
    "⇒": "entonces",
}
_ROOTS = {
    "√": "la raíz cuadrada de",
    "∛": "la raíz cúbica de",
    "∜": "la raíz cuarta de",
    ABS: "el valor absoluto de",
}
_FUNCTIONS = {
    "sen": "seno",
    "sin": "seno",
    "cos": "coseno",
    "tan": "tangente",
    "tg": "tangente",
    "cot": "cotangente",
    "sec": "secante",
    "csc": "cosecante",
    "log": "logaritmo",
    "ln": "logaritmo natural",
}
_CURRENCIES = {"dólar": ("dólar", "dólares"), "peso": ("peso", "pesos")}
# (singular, plural) after a number; only length units take "cuadrados" / "cúbicos".
_UNIT_WORDS = {
    "km": ("kilómetro", "kilómetros"),
    "cm": ("centímetro", "centímetros"),
    "mm": ("milímetro", "milímetros"),
    "m": ("metro", "metros"),
    "kg": ("kilogramo", "kilogramos"),
    "g": ("gramo", "gramos"),
    "mg": ("miligramo", "miligramos"),
    "ml": ("mililitro", "mililitros"),
    "mL": ("mililitro", "mililitros"),
    "L": ("litro", "litros"),
    "µm": ("micrómetro", "micrómetros"),
    "μm": ("micrómetro", "micrómetros"),
    "µL": ("microlitro", "microlitros"),
    "μL": ("microlitro", "microlitros"),
    "µl": ("microlitro", "microlitros"),
    "μl": ("microlitro", "microlitros"),
    "µg": ("microgramo", "microgramos"),
    "μg": ("microgramo", "microgramos"),
    "min": ("minuto", "minutos"),
    "h": ("hora", "horas"),
}
_LENGTH_UNITS = frozenset(["km", "cm", "mm", "m", "µm", "μm"])
_TIME_UNITS = frozenset(["h", "hr", "s", "seg", "min"])
# The unit after "/" ("km/h", "células/mL", "$15/kg"); prices and doses say "por kilo".
_PER_UNIT = {
    "h": "hora",
    "hr": "hora",
    "s": "segundo",
    "seg": "segundo",
    "min": "minuto",
    "km": "kilómetro",
    "cm": "centímetro",
    "mm": "milímetro",
    "m": "metro",
    "kg": "kilo",
    "g": "gramo",
    "mg": "miligramo",
    "mL": "mililitro",
    "ml": "mililitro",
    "L": "litro",
    "µL": "microlitro",
    "μL": "microlitro",
}


def _word_set(words: str) -> frozenset[str]:
    return frozenset(words.split())


_PER_WORDS = _word_set(
    """
    segundo minuto hora día semana mes año kilo kilogramo gramo litro mililitro metro kilómetro centímetro persona
    pieza unidad habitante alumno muestra
    """
)


def _with_plurals(words: str) -> frozenset[str]:
    out = set()
    for word in words.split():
        out.add(word)
        if word[-1] in "aeiouáéó":
            out.add(word + "s")
        elif word.endswith("z"):
            out.add(word[:-1] + "ces")
        elif word.endswith(("ón", "ús")):
            out.add(word[:-2] + word[-2].translate(_UNACCENT) + word[-1] + "es")
        else:
            out.add(word + "es")
    return frozenset(out)


# Agreement is opt-in: "un paso" / "una vez" / "doscientas manzanas" only before a known classroom noun (or the
# "solo" / "sola" in front of one). Any other word keeps the plain contract form ("tres más uno da cuatro",
# "x es igual a uno entonces"), so a guess can never turn a correct reading ungrammatical.
_MASC_NOUNS = _with_plurals(
    """
    paso lado número problema intento ejemplo ejercicio minuto segundo día año mes alumno niño chico maestro profesor
    amigo hermano punto metro kilómetro centímetro milímetro kilo kilogramo gramo miligramo litro mililitro galón peso
    dólar centavo grado ángulo término factor valor resultado libro cuaderno lápiz vaso plato pastel dulce chocolate
    caramelo boleto carro coche camión autobús perro gato pájaro árbol huevo entero decimal tercio cuarto medio cubo
    cuadrado triángulo círculo rectángulo vértice renglón capítulo tema sistema millón lugar semestre bimestre trimestre
    piso nivel grupo equipo juego partido gol voto salón organismo microorganismo habitante estudiante trozo pedazo
    billete bloque objeto elemento dato caso intervalo polígono cateto coeficiente exponente denominador numerador
    divisor múltiplo conjunto único último nuevo error acierto signo cuadro dibujo quiste trofozoíto microscopio río
    lago animal color cilindro prisma cono hexágono pentágono dálmata ciclo cultivo tubo pozo micrómetro microlitro
    frasco matraz filtro
    """
) | frozenset(["mil", "paréntesis"])
_FEM_NOUNS = _with_plurals(
    """
    vez manzana naranja pera fresa uva galleta canica moneda pregunta respuesta hora persona rebanada taza cucharada
    cucharadita pizza página hoja línea recta figura arista esquina caja bolsa botella pelota tarea semana clase parte
    suma resta alumna niña chica maestra profesora amiga hermana decena centena docena gráfica tabla columna fila letra
    cifra regla flor casa calle cuadra milla libra onza tonelada pulgada yarda gota papa tortilla paleta vuelta ronda
    manera medida variable incógnita ficha carta célula bacteria amiba especie planta semilla hormiga abeja mesa silla
    raíz base mano noche fase luz ley red diagonal sola única última nueva pista duda calculadora muestra laguna cara
    forma idea cosa potencia décima centésima tortuga serie pieza placa colonia micra tableta pastilla cápsula pipeta
    jeringa
    """
) | frozenset(["imagen", "imágenes"])
_FEM_SUFFIXES = ("ción", "ciones", "sión", "siones", "dad", "dades", "tad", "tades", "tud", "tudes", "umbre", "umbres")
# Nouns that are also verbs: "La figura 1 muestra que...", "1 suma 4" never agree.
_VERB_NOUNS = _with_plurals("muestra forma suma resta duda potencia parte cuadra")
_VERB_OBJECT = _word_set("que el la los las lo un una unos unas cómo como al parte partes")
# "solo" is also the adverb ("vale 1 solo cuando..."): it agrees only in front of a noun ("1 solo paso").
_SOLO = frozenset(["solo", "solos"])
# A plain noun after an unspaced "3–5" makes it a range ("3–5 muestras", "24–48 horas"), not a subtraction.
_RANGE_UNITS = _word_set("km cm mm m kg g mg ml mL L µm μm µL μL µl μl µg μg h hr hrs min seg s")
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
# "Calcula 8–3", "¿cuánto es 8-3?", "el resultado de 8-3": a subtraction even without another operator.
_SUBTRACT_CUES = _word_set(
    """
    calcula calcular calcule calculen calculamos resuelve resolver resuelva resuelvan evalúa evaluar evalúe simplifica
    simplificar efectúa efectuar opera realiza resta restar cuánto cuanto resultado haz intenta
    """
)
# Words that make "8-3 es 5" a statement about a subtraction.
_EQUALS_WORD = re.compile(
    r"[ \t]+(?:es|da|son|queda|vale|equivale(?:[ \t]+a)?|resulta(?:[ \t]+en)?|nos[ \t]+da|igual[ \t]+a)[ \t]+-?\d"
)
# fmt: off
_KEEP_RUNS = frozenset([
    "de", "del", "y", "e", "o", "u", "en", "a", "al", "es", "son", "era", "más", "por", "que", "con", "sin", "la",
    "las", "el", "los", "lo", "le", "se", "no", "sí", "si", "ni", "ya", "cm", "mm", "km", "kg", "mg", "ml", "cl",
    "dl", "hr", "hrs", "min", "seg", "lb", "lbs", "oz", "ft", "mi", "gal", "mph", "am", "pm", "un", "una", "uno",
    "tu", "su", "dos", "mil", "par", "muy", "hay", "ser", "fue", "van", "ver", "hoy", "ese", "esa", "eso", "pi",
    "usa", "les", "nos", "mis", "tus", "sus", "va", "da", "dan", "das", "te", "me", "yo", "ti", "ha", "han", "has",
    "haz", "pon", "ven",
])
# fmt: on

_NUM = r"\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?"
_COORD = r"-?[ \t]*(?:\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z])"
_SYMBOLS = "^/*=+<>×·÷√∛∜≤≥≠≈±%π∞→⇒⁄∕⁰¹²³⁴⁵⁶⁷⁸⁹⁻∈∉" + "".join(_UNICODE_FRACTIONS)
_MATH_HINT = re.compile(
    r"[\d" + _SYMBOLS + r"]|-(?=\d)|\|[^|\n]+\||(?<![^\W\d_])[a-zA-Z][ \t]*-[ \t]*[a-zA-Z(](?![^\W\d_])"
)
_MAYBE_MATH = re.compile(
    r"[\d" + _SYMBOLS + r"(\-|–—‒$]|(?<![^\W\d_])(?:[yY]|sen|cos|tg|cot|sec|csc|log|ln)(?![^\W\d_])"
)
_FN_NAMES = "sen|sin|cos|tan|tg|cot|sec|csc|log|ln"
_NORMALIZE = (
    (re.compile("[\ue000-\uf8ff]"), ""),
    (re.compile("[−﹣－‐‑]"), "-"),
    (re.compile("[⁄∕]"), "/"),
    (re.compile("<="), "≤"),
    (re.compile(">="), "≥"),
    (re.compile("!=|=/="), "≠"),
    (re.compile("={2,3}"), "="),
    (re.compile(r"(?<=[\w)])\*\*(?=[\w(])"), "^"),
    # "2 ** 3" is a power; "**3** y" is markdown bold, told apart by the symmetric spacing.
    (
        re.compile(
            r"(?:(?<=[\d)])|(?<![^\W\d_][a-zA-Z])(?<=[a-zA-Z]))([ \t]+)\*\*\1(?=[\d(]|-\d|[a-zA-Z](?![^\W\d_]))"
        ),
        "^",
    ),
    (re.compile(r"\*\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*\*"), r"\1"),
    (re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])"), r"\1"),
    (re.compile(r"\bsqrt\s*(?=\()", re.I), "√"),
    (re.compile(r"\bcbrt\s*(?=\()", re.I), "∛"),
    (re.compile(r"\by/o\b"), "y o"),
    (re.compile(r"(?<![\w/])c/u(?![\w/])", re.I), "cada uno"),  # "$5 c/u" in word problems
    (re.compile(r"\b(ra[ií]z)(?:\s+cuadrada)?(?:\s+de)?\s*√", re.I), r"\1 cuadrada de "),
    (re.compile(r"(?<![\w&#])#(?=\d)"), "número "),
    # "$2x$" is LaTeX inline math, not money; "$5 y $10" is money (a closing "$" never sits before a digit).
    (re.compile(r"(?<![\w$])\$(?=[^\s$\d]|\d+(?:\.\d+)?[a-zA-Z^({\\])([^$\n]*?[^\s$])\$(?![\w$])"), r"\1"),
    # An unclosed "$2x" opens LaTeX too: an amount glued to a lone variable is never money.
    (re.compile(r"(?<![\w$])\$(?=\d+(?:\.\d+)?[a-zA-Z](?![^\W\d_]))"), ""),
    # "3 x 4 = 12", "3 m x 5 m", "3x10^5": the letter x between numbers is a times sign.
    (re.compile(r"(\d(?:\.\d+)?(?:[ \t]*(?:km|cm|mm|m))?)[ \t]+[xX][ \t]+(?=\d)"), r"\1 × "),
    (re.compile(r"(?<=\d)[xX](?=10[ \t]*\^)"), "×"),
)
# SEP / NOM-008 group thousands with a space, a no-break space or a narrow no-break space: "25 000" is one number.
# A number right before or after the run makes it a list instead ("5 10 100", "10 100 1000").
_GROUP_SPACE = " " + chr(0x00A0) + chr(0x202F)
_SPACE_THOUSANDS = re.compile(
    rf"(?<![\d.,^/])(?<!\d[{_GROUP_SPACE}])(\d{{1,3}})((?:[{_GROUP_SPACE}]\d{{3}})+)(?!\d|[.,]\d{{3}}|[{_GROUP_SPACE}]\d)"
)
_NUMBER_RANGE = re.compile(rf"(?<![\w.,^/])({_NUM})([-–—‒])({_NUM})(?![^\W\d_µμ]|\d|[.,]\d|[\^/(])")
_BOUNDARY = re.compile(r"[!?;\n]|\.(?!\d)")
_SENTENCE_SPLIT = re.compile(r"((?:[!?;\n]|\.(?!\d))+)")
_SENTENCE_OPS = re.compile(r"[=<>≤≥≠≈+*/×·÷±^]|[-–—‒](?=[ \t]*(?:\d|\(|[a-zA-Z](?![^\W\d_])))")
_RANGE_TAIL = re.compile(rf"[ \t]*(?:[°%]|(?P<word>[^\W\d_]+)(?:[ \t]+(?P<month>{_MONTHS})(?![^\W\d_]))?)")
_INCLUSIVE_SLASH = re.compile(r"(?<![\w/])([^\W\d_]{2,})/(as|os|a|o|es|s)(?![\w/])")
_INCLUSIVE_PAREN = re.compile(r"(?<![\w(])([^\W\d_]{2,})\((as|os|a|o|es|s)\)(s?)(?![^\W\d_])")
_RANGE = re.compile(
    r"(?<![^\W\d_])((?:p[áa]gs?\.?|p[áa]ginas?|ejercicios?|problemas?|preguntas?|cap[íi]tulos?|incisos?|l[íi]neas?"
    r"|renglones|diapositivas?|lecciones|unidades|temas|pasos|secciones)[ \t]+)(\d+)[ \t]*[-–—‒][ \t]*(\d+)"
    r"(?![\d.,]|[ \t]*[-+*/=^<>])",
    re.I,
)
_DASH = re.compile("([ \t]*)[–—‒]([ \t]*)")
_LATER_OPERATOR = re.compile(r"[=+*/×·÷±<>≤≥≠≈^]|[-–‒](?=[ \t]*[\d(])")
# What may follow "–5" for the dash to be a sign: a degree, a percent, the end of the clause or another number.
_SIGNED_TAIL = re.compile(
    r"\d+(?:[.,]\d+)*[ \t]*(?:[°%]|grados?(?![^\W\d_])|(?:y|o|a|al|hasta)[ \t]+[-–‒]?\d|[.,;:!?)\]]|$)", re.M
)
_SUPERSCRIPT_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻ⁿ]+")
_UNICODE_FRACTION = re.compile("(\\d?)([" + "".join(_UNICODE_FRACTIONS) + "])")
_CURRENCY = re.compile(
    r"\$[ \t]?(?P<neg>-[ \t]?)?(?P<whole>\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.(?P<frac>\d+))?(?![^\W\d_])"
    r"(?:[ \t]+(?P<mag>mil(?:[ \t]+millones)?|(?:mill[óo]n|millones)(?:[ \t]+y[ \t]+medio)?)(?![^\W\d_]))?"
    r"(?:[ \t]+(?:de[ \t]+)?(?P<named>pesos?|d[óo]lar(?:es)?|dlls?|usd|mxn)(?![^\W\d_]))?",
    re.I,
)
# "$50 pesos" / "$50 MXN" / "$50 USD": a currency written after the amount wins over the default.
_NAMED_CURRENCY = {"peso": "peso", "pesos": "peso", "mxn": "peso"}
_UNIT = re.compile(
    rf"(?<![\w.,])(?:(?P<num>{_NUM})(?P<sp>[ \t]*))?(?<![^\W\d_])"
    r"(?P<unit>[µμ][mLlg]|km|cm|mm|kg|mg|ml|mL|min|m|g|L|h)(?:\^(?P<pow>[23]))?"
    r"(?:/(?P<per>hr|h|seg|s|min|km|cm|mm|m|kg|mg|g|mL|ml|L|[µμ][Ll])(?:\^(?P<perpow>[23]))?)?(?![^\W\d_]|\^|/)"
)
# "3 m por 5 m", "1.5 m de profundidad", "1 L de leche": a one-letter unit in a measurement, not a variable.
_UNIT_WORD_AFTER = re.compile(r"[ \t]+(?:de|por|x)(?![^\W\d_])|[ \t]*×")
_UNIT_WORD_BEFORE = re.compile(r"\d[ \t]*(?:km|cm|mm|m|kg|g|mg|L|mL|ml)(?:[ \t]+(?:por|x)[ \t]+|[ \t]*×[ \t]*)$")
# "5-10 g" (now "5 a 10 g"), "de 2 a 3 m", "entre 2 y 3 m": the end of a range carries the unit, not a variable.
_UNIT_RANGE_BEFORE = re.compile(rf"(?:\d[ \t]+al?|(?<![^\W\d_])entre[ \t]+(?:{_NUM})[ \t]+y)[ \t]+$", re.I)
_OPERATOR_AFTER = re.compile(r"[ \t]*[-–+*/×·÷=<>≤≥≠≈^(]")
# "células/mL", "metros/segundo", "$15/kg": a word or an amount over a unit is a rate, read "por".
_RATE = re.compile(
    "(?<![^\\W\\d_])(?P<left>[^\\W\\d_]+|" + OE + r")[ \t]*/[ \t]*(?P<den>[^\W\d_]+)(?:\^(?P<pow>[23]))?"
    r"(?![^\W\d_]|/|\^)"
)
_TIME = re.compile(
    r"(?P<pre>(?<![^\W\d_])(?:a[ \t]+)?las?[ \t]+)?(?<![\d:.,/])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?![\d:/]|,\d{3})"
    r"(?P<post>[ \t]*(?:[ap]\.?[ \t]?m\b\.?|hrs?\b|h\b|horas\b))?",
    re.I,
)
# Without "a las" / "h", "10:30" is still a time unless a ratio word or an equation is around.
_RATIO_CUE = re.compile(
    r"(?<![^\W\d_])(?:raz[óo]n|razones|escalas?|proporci[óo]n|proporciones|proporcional|relaci[óo]n|diluci[óo]n"
    r"|diluye|diluir|diluido|diluida|mezcla|simplifica|simplificar|equivalentes?)(?![^\W\d_])|=|::",
    re.I,
)
_TIME_CUE = re.compile(
    r"(?<![^\W\d_])(?:empieza|empiezan|comienza|comienzan|inicia|inician|termina|terminan|acaba|sale|salen|llega"
    r"|llegan|abre|cierra|dura|dur[óo]|tarda|tard[óo]|demora|desde|hasta)[ \t]+$",
    re.I,
)
_QUARTER_MINUTES = frozenset(["00", "15", "30", "45"])
# "a la 1" / "es la 1" is one o'clock, "la una"; "x elevado a la 1" is still a power.
_ONE_OCLOCK = re.compile(r"(?<![^\W\d_])((?:a|es)[ \t]+la[ \t]+)1(?!\d|[.,]\d|[:/^]|[ \t]*[-–—‒+*/×÷=<>^])", re.I)
_POWER_BEFORE = re.compile(r"(?:(?<![^\W\d_])(?:elevad[oa]s?|elevar|eleva|potencia|[a-zA-Z])|\d|\))[ \t]+$", re.I)
_RATIO_TERM = rf"(?:{_NUM})"
_PROPORTION = re.compile(
    rf"(?<![\w.,:/])({_RATIO_TERM})( ?):\2({_RATIO_TERM})[ \t]*(?:=|::)[ \t]*({_RATIO_TERM})( ?):\5({_RATIO_TERM})"
    r"(?![\w:/]|\.\d)"
)
_RATIO = re.compile(rf"(?<![\w.,:/])({_RATIO_TERM})(?:( ?):\2{_RATIO_TERM})+(?![\w:/]|\.\d)")
_DIVISION_AFTER = re.compile(rf"[ \t]*=[ \t]*{_RATIO_TERM}(?![\w:/]|\.\d)")
_INTERVAL = re.compile(
    r"(?<![^\W\d_])(?P<word>intervalos?(?:[ \t]+(?:abiertos?|cerrados?|semiabiertos?))?)[ \t]*"
    r"[\[(][ \t]*(?P<a>-?[ \t]*(?:\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z]|∞))[ \t]*,"
    r"[ \t]*(?P<b>-?[ \t]*(?:\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z]|∞))[ \t]*[\])]",
    re.I,
)
_ENDPOINT = r"-?[ \t]*(?:\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z]|∞)"
# "[2, 5]", "(2, 5]", "[0, ∞)": a square bracket makes the pair an interval; "(2, 5)" alone stays a point.
_BARE_INTERVAL = re.compile(
    rf"(?<![^\W\d_])(?P<open>[\[(])[ \t]*(?P<a>{_ENDPOINT})[ \t]*,[ \t]*(?P<b>{_ENDPOINT})[ \t]*(?P<close>[\])])"
)
_ELEMENT = r"-?[ \t]*(?:\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z])"
_SET = re.compile(
    rf"(?P<pre>(?<![^\W\d_])conjuntos?[ \t]*)?\{{[ \t]*(?P<items>{_ELEMENT}(?:[ \t]*,[ \t]*{_ELEMENT})+)"
    r"(?P<more>[ \t]*,[ \t]*(?:\.\.\.|…))?[ \t]*\}",
    re.I,
)
# "Los puntos (1, 2) y (3, 4)": the list already says "puntos", so later points need no "el punto".
_POINT_LIST = re.compile(
    r"(?<![^\W\d_])puntos[ \t]*\([^()\n]*\)(?:[ \t]*,[ \t]*\([^()\n]*\))*[ \t]*(?:,|y|e)[ \t]*$", re.I
)
_NAMED_POINT = re.compile(
    rf"(?<![^\W\d_])(?P<name>[A-Z]|[fgh])(?P<sp>[ \t]?)\([ \t]*(?P<a>{_COORD})[ \t]*,[ \t]*(?P<b>{_COORD})[ \t]*\)"
)
_POINT = re.compile(
    rf"(?P<pre>\b(?:punto|puntos|coordenadas?)[ \t]*)?\([ \t]*(?P<a>{_COORD})[ \t]*,[ \t]*(?P<b>{_COORD})[ \t]*\)",
    re.I,
)
_ABS_BARS = r"\|(?=[^\s|])([^()\[\]{}|\n]*?[^\s|])\|"
_GROUP = re.compile(
    r"\(([^()\[\]{}|]*)\)|\[([^()\[\]{}|]*)\]|\{([^()\[\]{}|]*)\}|" + _ABS_BARS,
)
_FUNCTION_CALL = re.compile(r"(?<![^\W\d_])(?:(sen|cos|tg|cot|sec|csc|log|ln)[ \t]*|(sin|tan))(?=\()", re.I)
_FUNCTION_POWER = re.compile(rf"(?<![^\W\d_])({_FN_NAMES})\^([23])[ \t]*(?=\()", re.I)
_FUNCTION_WORD = re.compile(rf"(?<![^\W\d_])({_FN_NAMES}){FN}(?={LF})", re.I)
_BARE_FUNCTION = re.compile(
    r"(?<![^\W\d_])(sen|cos|tg|cot|sec|csc|log|ln)[ \t]+(?=\d|[a-zA-Z](?![^\W\d_]))",
    re.I,
)
_MATH_CONTENT = re.compile("[\\s\\d.a-zA-Z+\\-*/×·÷=<>≤≥≠≈±^√∛∜%π°\ue000-\ue00d]*")
_TERM = re.compile(r"\s*-?\s*(?:\d+(?:\.\d+)?(?:/\d+)?[a-zA-Z]?|[a-zA-Z]|π)\s*")
_BINARY = re.compile("[\\w).%π" + _CLOSERS + "]\\s*[+\\-*/×·÷=<>≤≥≠≈±]")
_NESTED = re.compile("[\ue000\ue002\ue004\ue006][^\ue000-\ue007]*[\ue001\ue003\ue005\ue007]")
_INT = re.compile(r"\s*-?\s*\d+\s*")
_NO_OPERAND_BEFORE = "(?<![\\w.,/^" + _CLOSERS + "])"
_MIXED = re.compile(_NO_OPERAND_BEFORE + r"(\d+)[ \t]+(\d+)/(\d+)(?![\d/^]|\.\d)")
_FRACTION = re.compile(_NO_OPERAND_BEFORE + r"(\d+)/(\d+)(?![\d/^]|\.\d)")
_EXPONENT = re.compile(
    rf"([{RQ}{RS}])?[ \t]*\^[ \t]*(?:(-?)[ \t]*(\d+)(?!\d|\.\d)|(-?)[ \t]*([a-zA-Z])(?![a-zA-Z])|({LE}))?"
)
_GROUP_AFTER_OPERAND = re.compile(
    "(?<=[\\d" + _CLOSERS + YV + "])(?=[" + LQ + LS + _PREFIX + "])"
    "|(?<![a-zA-Z][a-zA-Z])(?<=[a-zA-Z])(?=[" + LQ + LS + _PREFIX + "])"
)
_OPERAND_AFTER_GROUP = re.compile("(?<=[" + RQ + RS + "])(?=[\\da-zA-Z" + OS + YV + "])")
_ORDINAL_SIGN = re.compile(r"(?<![\w.,])(10|[1-9])\.?([ºª])")
_ORDINAL_SUFFIX = re.compile(r"(?<![\w.,])(10|[1-9])(er|ero|era|do|da|ro|ra|to|ta|vo|va|no|na|mo|ma)(?![^\W\d_])")
# In Mexico "3° de secundaria" / "1° grado" are ordinals, not angles.
_ORDINAL_DEGREE = re.compile(
    r"(?<![\w.,])(10|[1-9])[ \t]*°(?=[ \t]+(de[ \t]+(?:secundaria|primaria|preparatoria|prepa|bachillerato|k[íi]nder)"
    rf"|de[ \t]+(?:{_MONTHS}|cada[ \t]+mes)|del[ \t]+mes|grado|lugar|año|semestre|bimestre|trimestre|cuatrimestre"
    r"|periodo|período|parcial|piso|nivel|grupo)(?![^\W\d_]))",
    re.I,
)
_MONTH_AFTER = re.compile(rf"de[ \t]+(?:{_MONTHS}|cada[ \t]+mes)|del[ \t]+mes", re.I)
# "Tu calificación es 10/10", "Contestaste 7/10 bien": a score is "diez de diez", never "diez décimos".
_SCORE_BEFORE = re.compile(
    r"(?<![^\W\d_])(?:calificaci[óo]n|calificaste|sacaste|sac[óo]|saqu[ée]|obtuviste|obtuvo|contestaste|acertaste"
    r"|puntuaci[óo]n|puntaje|nota|aciertos)(?![^\W\d_])[^.;!?\n\d]{0,20}$",
    re.I,
)
_SCORE_AFTER = _word_set("bien correctas correctos aciertos buenas buenos puntos")
_PERCENT = re.compile(rf"(?<![\d.,])({_NUM})[ \t]*%")
_DEGREES = re.compile(rf"(?<![\d.,])({_NUM})[ \t]*°(?:[ \t]*([CF])(?![a-zA-Z]))?")
_OPERATOR = re.compile("->|=>|[-+=<>≤≥≠≈±×÷/*·→⇒]")
_NUMBER = re.compile(r"(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.(\d+))?")
_NEXT_WORD = re.compile(r"[ \t]+([^\W\d_]{2,})")
# "1 a 2 tazas" / "1 o 2 tazas": the noun after the range sets the gender of both numbers.
_RANGE_NEXT = re.compile(
    r"[ \t]+(?:a|al|y|o)[ \t]+(?:\d+(?:\.\d+)?[ \t]+([^\W\d_]{2,})"
    + "|"
    + OS
    + SOFT
    + "[^"
    + OS
    + OE
    + "]*?([^\\W\\d_]{2,})"
    + SOFT
    + OE
    + ")"
)
# "Al 21 suma 4": a "noun" followed by an operand is a verb. The vowels are Spanish words ("suma a", "resta y").
_NOUN_IS_VERB = re.compile(r"[ \t]+(?:-?[ \t]*\d|[b-df-hj-np-tv-xzB-DF-HJ-NP-TV-XZ](?![^\W\d_]))")
_LETTER_RUN = re.compile(r"(?<![^\W\d_])[a-z]{2,3}(?![^\W\d_])")
_VARIABLE_Y = re.compile(r"(?<![^\W\d_])[yY](?![^\W\d_])")
_PREV_WORD = re.compile(r"([^\W\d_]+)[ \t]+$")
# No conjunction can follow these words: "despeja y", "el eje y", "el valor de y", "despeja la y", "x y y".
_Y_CUES = frozenset(["despeja", "despejar", "despejamos", "despejas", "eje", "variable", "de", "la", "y"])
_Y_DESPEJA = frozenset(["despeja", "despejar", "despejamos", "despejas"])
# "Despeja y calcula la cantidad": a second instruction right after "despeja y" makes the y a conjunction.
_Y_SECOND_VERB = _word_set(
    """
    calcula comprueba verifica sustituye reemplaza resuelve simplifica encuentra halla escribe anota grafica evalúa
    factoriza multiplica divide suma resta explica responde contesta dime determina obtén revisa compara redondea
    aplica interpreta identifica completa justifica expresa reduce ordena traza dibuja marca mide registra calcular
    comprobar verificar sustituir reemplazar resolver simplificar encontrar hallar escribir anotar graficar evaluar
    factorizar multiplicar dividir sumar restar explicar responder contestar determinar obtener revisar comparar
    """
)
_Y_VERBS = frozenset(
    ["encuentra", "encontrar", "halla", "hallar", "calcula", "calcular", "determina", "determinar", "obtén", "obtener"]
)
# After "encuentra" / "calcula", y is the variable only before a math cue ("Encuentra y en 2x + y = 5");
# any other word ("Calcula y luego suma", "Encuentra y dime") keeps the conjunction.
_Y_MATH_NEXT = re.compile(
    r"[ \t]+(?:(?:en|sabiendo|dad[oa]s?|tal[ \t]+que|a[ \t]+partir)(?![^\W\d_])"
    r"|(?:si|cuando|para)[ \t]+(?:-?\d|[a-zA-Z](?![^\W\d_]))"
    r"|de[ \t]+(?!(?:ah[íi]|aqu[íi]|all[íi]|nuevo|inmediato|igual|manera|forma|modo|una)(?![^\W\d_])))",
    re.I,
)
# "Resuelve para y" names the variable; "Para y piensa" is a conjunction.
_Y_SOLVE = re.compile(
    r"(?<![^\W\d_])(?:resuelve|resolver|resolvemos|resuelva|resuelvan|despeja|despejar|despejamos|soluciona"
    r"|solucionar)(?![^\W\d_])[^.;:!?\n]*$",
    re.I,
)
_CONTRACT = re.compile("(?<![^\\W\\d_])(a|de)([ \t" + OS + OE + SOFT + "]*)" + EL, re.I)
_SOFT_RUN = re.compile("[ \t]*" + SOFT + "[ \t" + SOFT + "]*")
_NESTED_QUANTITY = re.compile(LQ + "[ \t" + SOFT + "]*(?=" + LQ + ")")


@functools.cache
def _shared_detector() -> Callable[[str], bool] | None:
    # One detector for both languages once the English module lands; this copy covers only its absence.
    try:
        module = importlib.import_module("biobench_sar.llm.mathspeak")
    except ImportError:
        return None
    detector = getattr(module, "has_math_symbols", None)
    return detector if callable(detector) else None


def has_math_symbols(text: str) -> bool:
    """True when `text` holds math a Spanish voice would misread: the shared detector plus Spanish hints."""
    if not text:
        return False
    shared = _shared_detector()
    return bool(_MATH_HINT.search(text)) or (shared is not None and bool(shared(text)))


def number_to_words_es(n: int, *, fem: bool = False, apocope: bool = False) -> str:
    """Cardinal in words; `apocope` for a following masculine noun ("un", "veintiún"), `fem` for a feminine one."""
    if n < 0:
        return "menos " + number_to_words_es(-n, fem=fem, apocope=apocope)
    if n == 0:
        return "cero"
    if n >= 10**12:
        return " ".join(_UNITS[int(d)] for d in str(n))
    millions, rest = divmod(n, 10**6)
    thousands, units = divmod(rest, 1000)
    parts = []
    if millions:
        parts.append("un millón" if millions == 1 else number_to_words_es(millions, apocope=True) + " millones")
    if thousands:
        parts.append("mil" if thousands == 1 else _below_thousand(thousands, fem, True) + " mil")
    if units:
        parts.append(_below_thousand(units, fem, apocope))
    return " ".join(parts)


def fraction_to_words_es(num: int, den: int) -> str:
    name = _denominator(den)
    if name is None:
        return f"{number_to_words_es(num)} sobre {number_to_words_es(den)}"
    return f"{number_to_words_es(num, apocope=True)} {name if num == 1 else name + 's'}"


def to_speech_es(text: str, *, currency: str = "dólar") -> str:
    """Math in `text` spoken as es-MX words; `currency` ("dólar" or "peso") names what "$" counts."""
    if currency not in _CURRENCIES:
        raise ValueError(f"currency must be one of {sorted(_CURRENCIES)}")
    if not text or not _MAYBE_MATH.search(text):
        return text
    try:
        return _speak(text, _CURRENCIES[currency])
    except Exception:  # the TTS thread must never die on odd input: the raw text is still speakable
        return text


def _speak(text: str, currency: tuple[str, str]) -> str:
    s = _normalize(text)
    s = _CURRENCY.sub(lambda m: _money(m, currency), s)
    s = _RATE.sub(_rate, s)
    s = _TIME.sub(_time, s)
    s = _ONE_OCLOCK.sub(_one_oclock, s)
    s = _PROPORTION.sub(_proportion, s)
    s = _RATIO.sub(_ratio, s)
    s = _ORDINAL_SIGN.sub(_ordinal, s)
    s = _ORDINAL_SUFFIX.sub(_ordinal, s)
    s = _ORDINAL_DEGREE.sub(_ordinal_degree, s)
    s = _UNIT.sub(_unit, s)
    s = _INTERVAL.sub(_interval, s)
    s = _BARE_INTERVAL.sub(_bare_interval, s)
    s = _SET.sub(_set, s)
    s = _NAMED_POINT.sub(_named_point, s)
    s = _POINT.sub(_point, s)
    s = _FUNCTION_POWER.sub(lambda m: f"{_FUNCTIONS[m[1].lower()]} {'cuadrado' if m[2] == '2' else 'cúbico'}{FN}", s)
    s = _FUNCTION_CALL.sub(lambda m: (m[1] or m[2]) + FN, s)
    s = _mark_groups(s)
    s = _FUNCTION_WORD.sub(lambda m: _FUNCTIONS[m[1].lower()], s).replace(FN, "")
    s = _BARE_FUNCTION.sub(lambda m: _FUNCTIONS[m[1].lower()] + " de ", s)
    s = _LETTER_RUN.sub(_split_letters, s)
    s = _VARIABLE_Y.sub(_variable_y, s)
    s = s.replace("∉", SOFT + "no pertenece a" + SOFT).replace("∈", SOFT + "pertenece a" + SOFT)
    s = _MIXED.sub(_mixed, s)
    s = _FRACTION.sub(_fraction, s)
    s = _EXPONENT.sub(_exponent, s)
    s = s.replace("π", _chunk("pi")).replace("∞", _chunk("infinito"))
    s = _GROUP_AFTER_OPERAND.sub(TIMES, s)
    s = _OPERAND_AFTER_GROUP.sub(TIMES, s)
    for sign, words in _ROOTS.items():
        s = s.replace(sign, OS + SOFT + words + SOFT)
    s = _PERCENT.sub(lambda m: _chunk(_spell(m[1]) + " por ciento"), s)
    s = s.replace("%", _chunk("por ciento"))
    s = _DEGREES.sub(_degrees, s)
    s = _OPERATOR.sub(_operator, s)
    s = _NUMBER.sub(_number, s)
    return _finish(s)


def _below_thousand(n: int, fem: bool, apocope: bool) -> str:
    if n == 100:
        return "cien"
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        word = _HUNDREDS[hundreds]
        parts.append(word[:-2] + "as" if fem and hundreds > 1 else word)
    if rest:
        tens, unit = divmod(rest, 10)
        if rest < 30:
            parts.append(_one(_UNITS[rest], fem, apocope) if rest in (1, 21) else _UNITS[rest])
        else:
            parts.append(
                _TENS[tens] + (" y " + (_one("uno", fem, apocope) if unit == 1 else _UNITS[unit]) if unit else "")
            )
    return " ".join(parts)


def _one(word: str, fem: bool, apocope: bool) -> str:
    if fem:
        return word[:-1] + "a"
    if apocope:
        return "veintiún" if word == "veintiuno" else "un"
    return word


def _denominator(den: int) -> str | None:
    if den in _DENOMS:
        return _DENOMS[den]
    if 11 <= den <= 29:
        return _UNITS[den].translate(_UNACCENT) + "avo"
    if 30 <= den <= 90 and den % 10 == 0:
        return _TENS[den // 10][:-1] + "avo"
    return None


def _spell(raw: str, *, fem: bool = False, apocope: bool = False) -> str:
    whole, _, frac = raw.replace(",", "").partition(".")
    if len(whole) > _MAX_DIGITS:
        words = _digits(whole.lstrip("0") or "0")
    else:
        words = number_to_words_es(int(whole), fem=fem, apocope=apocope and not frac)
    if frac:
        # "3.14" -> "tres punto catorce" like a Mexican classroom; leading zeros / long tails go digit by digit.
        tail = number_to_words_es(int(frac)) if len(frac) <= 2 and not frac.startswith("0") else _digits(frac)
        words += " punto " + tail
    return words


def _digits(digits: str) -> str:
    return " ".join(_UNITS[int(d)] for d in digits)


def _chunk(words: str) -> str:
    return OS + SOFT + words + SOFT + OE


def _count(raw: str, singular: str, plural: str, *, fem: bool = False) -> str:
    words = _spell(raw, fem=fem, apocope=not fem)
    if words.endswith(("millón", "millones")):
        words += " de"
    return f"{words} {singular if raw == '1' else plural}"


def _normalize(text: str) -> str:
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    text = _INCLUSIVE_SLASH.sub(_inclusive, text)
    text = _INCLUSIVE_PAREN.sub(_inclusive, text)
    text = _UNICODE_FRACTION.sub(lambda m: (m[1] + " " if m[1] else "") + _UNICODE_FRACTIONS[m[2]], text)
    text = _SUPERSCRIPT_RUN.sub(lambda m: "^" + m[0].translate(_SUPERSCRIPTS), text)
    text = _SPACE_THOUSANDS.sub(_space_thousands, text)
    text = _RANGE.sub(r"\1\2 a \3", text)
    text = _number_ranges(text)
    return _DASH.sub(_dash, text)


def _space_thousands(m: re.Match[str]) -> str:
    head, groups = m[1], m[2]
    if " " in groups and len(head) == 3 and not groups.endswith("000"):
        return m[0]  # "100 200 300" written with plain spaces is a list
    return head + re.sub(f"[{_GROUP_SPACE}]", "", groups)


def _number_ranges(text: str) -> str:
    # One pass per sentence, so the "is anything else math here?" test stays linear on long input.
    if not _NUMBER_RANGE.search(text):
        return text
    parts = _SENTENCE_SPLIT.split(text)
    for k in range(0, len(parts), 2):
        found = len(_NUMBER_RANGE.findall(parts[k]))
        if found:
            busy = len(_SENTENCE_OPS.findall(parts[k])) > found
            parts[k] = _NUMBER_RANGE.sub(functools.partial(_number_range, busy=busy), parts[k])
    return "".join(parts)


def _number_range(m: re.Match[str], *, busy: bool) -> str:
    # "3–5 muestras", "25-37 °C", "1990–2000" are ranges; "8–3 = 5", "2(8-3)", "Calcula 8–3" subtract.
    s, i, j = m.string, m.start(), m.end()
    minus = m[1] + "-" + m[3]
    upward = float(m[1].replace(",", "")) < float(m[3].replace(",", ""))
    if _in_expression(s, i, j):
        prev = _prev_char(s, i)
        # "2-3 días = 48-72 horas": a measurement after a relation is still a range ("x = 3-5 m" may be algebra).
        measured = prev != "" and prev in _RELATIONS and not _joined_after(s, j) and _range_tail(s, j, short=False)
        if not (measured and upward):
            return minus
    if not _range_tail(s, j):
        words = re.findall(r"[^\W\d_]+", s[max(0, i - 32) : i])[-2:]
        if busy or any(w.lower() in _SUBTRACT_CUES for w in words):
            return minus
        if not upward:
            return minus  # a range runs upward: a bare "8-3" is a subtraction
    before = _PREV_WORD.search(s, max(0, i - 24), i)
    word = before[1].lower() if before else ""
    joiner = " y " if word == "entre" else " al " if word == "del" else " a "
    return m[1] + joiner + m[3]


def _in_expression(s: str, i: int, j: int) -> bool:
    k = i - 1
    while k >= 0 and s[k] in " \t":
        k -= 1
    if k >= 0 and s[k] in "([":
        if k > 0 and (s[k - 1].isalnum() or s[k - 1] == ")"):
            return True  # "2(8-3)"
        k -= 1
        while k >= 0 and s[k] in " \t":
            k -= 1
    if k >= 0 and s[k] in _OPS + _PREFIX + "–—‒|":
        return True  # "x = 8-3", "√(8-3)"
    return _joined_after(s, j)


def _joined_after(s: str, j: int) -> bool:
    k = j
    while k < len(s) and s[k] in " \t)]":
        k += 1
    if k < len(s) and s[k] in _OPS + "–—‒(|":
        return True
    if k < len(s) and _lone_letter_after(s, k) and s[k] not in "aeouyAEOUY" and s[k] not in _RANGE_UNITS:
        return True  # "3-5 x" is algebra
    return bool(_EQUALS_WORD.match(s, j))


def _range_tail(s: str, j: int, *, short: bool = True) -> bool:
    tail = _RANGE_TAIL.match(s, j)
    if not tail:
        return False
    word = tail["word"]
    if word is None:
        return True  # "25-37 °C", "5–10 %"
    if word.lower() == "de":
        return bool(tail["month"])  # "Del 3–7 de marzo"
    if word in _RANGE_UNITS:
        return short or len(word) > 1  # a one-letter unit ("m", "g") can also be a variable
    return _noun_gender(word) is not None or _noun_gender(word.lower()) is not None


def _other_gender(stem: str, suffix: str) -> str | None:
    low = stem.lower()
    if suffix in ("a", "o"):
        other = "o" if suffix == "a" else "a"
        if low.endswith(other):
            return stem[:-1] + suffix
        # "campeón/a" -> "campeona", "profesor/a" -> "profesora", "un/a" -> "una"
        if suffix == "a" and low[-1] in "nrlsz" and low[-2] in "aeiouáéíóú":
            return stem[:-2] + stem[-2].translate(_UNACCENT) + stem[-1] + "a"
    elif suffix in ("as", "os"):
        other = "os" if suffix == "as" else "as"
        if low.endswith(other):
            return stem[:-2] + suffix
        if suffix == "as" and low.endswith("es"):
            return stem[:-2] + "as"
    return None


def _inclusive(m: re.Match[str]) -> str:
    # "listo/a", "seguro(a)", "niño(a)s", "respuesta(s)": inclusive writing, never a quotient or a product.
    stem, suffix = m[1], m[2]
    tail = m[3] if m.re.groups == 3 else ""
    low = stem.lower()
    if low in _FUNCTIONS:
        return m[0]
    if suffix in ("s", "es"):
        optional_plural = (suffix == "s") == (low[-1] in "aeiouáéíóú")
        return stem + tail if optional_plural else m[0]
    other = _other_gender(stem, suffix)
    if other is not None:
        return f"{stem}{tail} o {other}{tail}"
    return stem + tail if len(stem) >= 4 else m[0]


def _lone_letter_before(s: str, i: int) -> bool:
    return i > 0 and s[i - 1].isascii() and s[i - 1].isalpha() and (i < 2 or not s[i - 2].isalpha())


def _lone_letter_after(s: str, j: int) -> bool:
    return j < len(s) and s[j].isascii() and s[j].isalpha() and (j + 1 >= len(s) or not s[j + 1].isalpha())


def _dash(m: re.Match[str]) -> str:
    # Word turns " - " into " – ": an en / em / figure dash between operands is a minus; prose dashes stay.
    s, i, j = m.string, m.start(), m.end()
    left, right = s[i - 1 : i], s[j : j + 1]
    left_letter, right_letter = _lone_letter_before(s, i), _lone_letter_after(s, j)
    left_operand = left.isdigit() or (left != "" and left in ")]π") or left_letter
    right_operand = right.isdigit() or (right != "" and right in "([√") or right_letter
    dash = s[i + len(m[1])]
    if m[1] == m[2] and left_operand and right_operand:
        if left_letter and right_letter and left.lower() in "aeouy" and right.lower() in "aeouy":
            return m[0]  # "y — o" are words
        if right_letter and (dash == "—" or right.lower() in "aeouy") and not _variable_ends(s, j + 1, dash):
            return m[0]  # "7 — y lo hiciste", "Paso 2 — a la derecha": an aside, not a minus
        return m[1] + "-" + m[2]
    if not m[2] and (right.isdigit() or (right != "" and right in "(√") or right_letter):
        prev = _prev_char(s, i)
        if prev and (prev in _RELATIONS + "+*/×·÷±^" or (dash != "—" and prev in "([")):
            return m[1] + "-"  # "x = –3", "x = –b"
        if dash != "—" and prev in ("", "\n"):
            line_end = s.find("\n", j)
            if _LATER_OPERATOR.search(s, j, len(s) if line_end < 0 else line_end):
                return m[1] + "-"  # "–5 + 3 = –2" opens with a negative number
        if dash != "—" and right.isdigit() and not _ends_operand(s, i) and _SIGNED_TAIL.match(s, j):
            return m[1] + "-"  # "Hace –5 °C", "–5, –3 y 0": a negative reading; "–5 minutos más, dijo" stays
    return m[0]


def _variable_ends(s: str, k: int, dash: str) -> bool:
    # A lone letter after a dash is a variable only where an expression can go on or stop ("x = 5 — y",
    # "(x – a)"); an em dash before "b) siguiente" is still prose.
    nxt = _next_char(s, k)
    return not nxt or nxt in _OPS + ".;:?!" + ("" if dash == "—" else ")]")


def _money(m: re.Match[str], currency: tuple[str, str]) -> str:
    whole, frac, named = m["whole"], m["frac"], (m["named"] or "").lower()
    if named:
        currency = _CURRENCIES[_NAMED_CURRENCY.get(named, "dólar")]
    singular, plural = currency
    sign = "menos " if m["neg"] else ""
    if m["mag"]:
        # "$10 mil" -> "diez mil dólares", "$1.5 millones" -> "uno punto cinco millones de dólares"
        magnitude = re.sub(r"\bmillon\b", "millón", " ".join(m["mag"].lower().split()))
        amount = _spell(whole + ("." + frac if frac else ""), apocope=True)
        words = magnitude if (amount, magnitude) == ("un", "mil") else f"{amount} {magnitude}"
        return _chunk(f"{sign}{words}{' ' if magnitude == 'mil' else ' de '}{plural}")
    if frac is not None and len(frac) > 2:
        return _chunk(f"{sign}{_spell(whole + '.' + frac)} {plural}")
    cents = frac.ljust(2, "0").lstrip("0") if frac else ""
    parts = []
    if whole.strip("0,") or not cents:
        parts.append(_count(whole, singular, plural))
    if cents:
        parts.append(_count(cents, "centavo", "centavos"))
    return _chunk(sign + " con ".join(parts))


def _rate(m: re.Match[str]) -> str:
    left, den, power = m["left"], m["den"], m["pow"]
    if left != OE and (len(left) < 4 or not left.endswith("s") or left[:4].lower() == den[:4].lower()):
        return m[0]  # a rate has a plural or an amount on top: "semana/mes" and "alumnas/alumno" are alternatives
    word = _PER_UNIT.get(den) or (den.lower() if den.lower() in _PER_WORDS else None)
    if word is None:
        return m[0]  # "sí/no", "verdadero/falso" are alternatives
    if power:
        if den in _LENGTH_UNITS:
            word += " cuadrado" if power == "2" else " cúbico"
        elif den in _TIME_UNITS and power == "2":
            word += " al cuadrado"
        else:
            return m[0]
    return left + _chunk("por " + word)


def _sentence_span(s: str, i: int, j: int, limit: int = 200) -> tuple[int, int]:
    start, end = max(0, i - limit), min(len(s), j + limit)
    for boundary in _BOUNDARY.finditer(s, start, i):
        start = boundary.end()
    after = _BOUNDARY.search(s, j, end)
    return start, after.start() if after else end


def _time(m: re.Match[str]) -> str:
    if not (m["pre"] or m["post"]):
        s, start = m.string, m.start()
        a, b = _sentence_span(s, start, m.end())
        if _RATIO_CUE.search(s, a, b):
            return m[0]  # "La razón es 3:45", "Diluye 1:10", "2:30 = 4:60"
        if m["m"] not in _QUARTER_MINUTES and not _TIME_CUE.search(s, max(0, start - 16), start):
            return m[0]
    hour = number_to_words_es(int(m["h"]), fem=True)
    minutes = "en punto" if m["m"] == "00" else "y " + number_to_words_es(int(m["m"]))
    return (m["pre"] or "") + _chunk(f"{hour} {minutes}") + (m["post"] or "")


def _one_oclock(m: re.Match[str]) -> str:
    if _POWER_BEFORE.search(m.string, max(0, m.start() - 16), m.start()):
        return m[0]
    return m[1] + _chunk("una")


def _proportion(m: re.Match[str]) -> str:
    a, b, c, d = (_spell(m[k]) for k in (1, 3, 4, 6))
    return _chunk(f"{a} es a {b} como {c} es a {d}")


def _ratio(m: re.Match[str]) -> str:
    terms = [_spell(t.strip()) for t in m[0].split(":")]
    # "8 : 2 = 4" divides; "2:3" (or "2:3:5") is a ratio, read "dos es a tres".
    joiner = " entre " if len(terms) == 2 and _DIVISION_AFTER.match(m.string, m.end()) else " es a "
    return _chunk(joiner.join(terms))


def _interval(m: re.Match[str]) -> str:
    a, b = _coordinate(m["a"]), _coordinate(m["b"])
    return f"{m['word']} {OS}{SOFT}de{SOFT}{a}{SOFT}a{SOFT}{b}{SOFT}{OE}"


def _bare_interval(m: re.Match[str]) -> str:
    if m["open"] == "(" and m["close"] == ")":
        return m[0]  # the point pass reads "(2, 5)"
    a, b = _coordinate(m["a"]), _coordinate(m["b"])
    return f"{OS}{SOFT}{EL}{SOFT}intervalo{SOFT}de{SOFT}{a}{SOFT}a{SOFT}{b}{SOFT}{OE}"


def _set(m: re.Match[str]) -> str:
    items = ", ".join(_coordinate(t) for t in m["items"].split(","))
    if m["more"]:
        items += ", etcétera"
    label = "" if m["pre"] else f"{EL}{SOFT}conjunto{SOFT}"
    return f"{m['pre'] or ''}{OS}{SOFT}{label}{items}{SOFT}{OE}"


def _named_point(m: re.Match[str]) -> str:
    name = m["name"]
    if m["sp"] and name in "EOUY":
        return m[0]  # "Y (2, 3)" opens a sentence with a conjunction
    label = "de" if name in "fgh" else "de coordenadas"
    return f"{name}{OS}{SOFT}{label}{SOFT}{_coordinate(m['a'])}{SOFT}, {_coordinate(m['b'])}{SOFT}{OE}"


def _point(m: re.Match[str]) -> str:
    pre = m["pre"] or ""
    listed = not pre and _POINT_LIST.search(m.string, max(0, m.start() - 200), m.start())
    label = "" if pre or listed else f"{EL}{SOFT}punto"
    return f"{pre}{OS}{SOFT}{label}{SOFT}{_coordinate(m['a'])}{SOFT}, {_coordinate(m['b'])}{SOFT}{OE}"


def _coordinate(raw: str) -> str:
    return re.sub(r"\s+", "", raw).replace("y", YV).replace("Y", YV)


def _unit(m: re.Match[str]) -> str:
    num, unit, power, per, perpow = m["num"], m["unit"], m["pow"], m["per"], m["perpow"]
    if len(unit) == 1 and not (num and m["sp"] and (power or per or _measurement(m))):
        return m[0]  # "3m" / "5 m" can be algebra; "12 m²", "9.8 m/s" and "1.5 m de" are units
    if (power and unit not in _LENGTH_UNITS) or (unit == "min" and not num):
        return m[0]
    if not num and not per:
        return m[0]
    singular, plural = _UNIT_WORDS[unit]
    words = _count(num, singular, plural, fem=unit == "h") if num else plural
    if power:
        adjective = "cuadrado" if power == "2" else "cúbico"
        words += " " + (adjective if num == "1" else adjective + "s")
    if per:
        words += " por " + _PER_UNIT[per]
        if perpow and per in _LENGTH_UNITS:
            words += " cuadrado" if perpow == "2" else " cúbico"  # "g/cm³"
        elif perpow == "2" and per in _TIME_UNITS:
            words += " al cuadrado"  # "m/s²"
        elif perpow:
            return m[0]
    return _chunk(words)


def _measurement(m: re.Match[str]) -> bool:
    s, unit = m.string, m["unit"]
    if unit == "h":
        return not _OPERATOR_AFTER.match(s, m.end())  # "24 h" is hours; "5 h + 2" stays algebra
    if "." in m["num"] or _UNIT_WORD_AFTER.match(s, m.end()):
        return True
    if _UNIT_WORD_BEFORE.search(s, max(0, m.start() - 16), m.start()):
        return True
    if _OPERATOR_AFTER.match(s, m.end()):
        return False  # "de 2 a 3 m + 1" is still algebra
    return bool(_UNIT_RANGE_BEFORE.search(s, max(0, m.start() - 32), m.start()))


def _prev_char(s: str, i: int) -> str:
    k = i - 1
    while k >= 0 and s[k] in " \t" + SOFT:
        k -= 1
    return s[k] if k >= 0 else ""


def _next_char(s: str, j: int) -> str:
    while j < len(s) and s[j] in " \t" + SOFT:
        j += 1
    return s[j] if j < len(s) else ""


def _split_letters(m: re.Match[str]) -> str:
    # "2ab" / "y = mx + b": a short letter run inside math is a product of variables, not a word.
    run, s = m[0], m.string
    if run in _KEEP_RUNS:
        return run
    prev, nxt = _prev_char(s, m.start()), _next_char(s, m.end())
    coefficient = m.start() > 0 and s[m.start() - 1].isdigit()
    framed = bool(prev) and prev in _OPS + _OPENERS + _PREFIX and (not nxt or nxt in _OPS + _CLOSERS + ".,;:?!")
    # "xy = 6" opens with a product of variables: no word stands at the start right before a relation.
    leading = prev in ("", "\n", ":", "¿", "¡") and bool(nxt) and nxt in _RELATIONS + "+*×·÷±^"
    if not (coefficient or framed or leading):
        return run
    return SOFT + SOFT.join(YV if c == "y" else c for c in run) + SOFT


def _variable_y(m: re.Match[str], *, chained: bool = True) -> str:
    s, i, y = m.string, m.start(), m[0]
    prev, nxt = _prev_char(s, i), _next_char(s, m.end())
    if (i > 0 and s[i - 1].isdigit()) or (prev and prev in _OPS + _OPENERS + _PREFIX + "%"):
        return YV
    if y == "Y":
        # A capital "Y" opens sentences; it is the variable only right before an operator ("Y = 2x").
        return YV if nxt and nxt in _OPS.replace("-", "") else y
    # "y - 3 = 0" is the variable, but "son 3 y -2" is the conjunction before a negative number.
    if nxt and (nxt in _OPS.replace("-", "") + _CLOSERS + "∈∉" or (nxt == "-" and not prev)):
        return YV
    # A conjunction cannot end a clause: "¿cuánto vale y?", "los ejes x, y" (but "Bueno, y." is still one).
    clause_end = (not nxt or nxt in "?.!;:") and bool(prev) and prev not in "¿¡"
    if clause_end and (prev != "," or _operand_before_comma(s, i)):
        return YV
    word = _PREV_WORD.search(s, max(0, i - 24), i)
    cue = word[1].lower() if word else ""
    if cue in _Y_CUES:
        if cue == "y" and chained:
            # "x y y" ends on the variable, but "el valor de y y el de x" joins the variable to more prose.
            before = _VARIABLE_Y.match(s, word.start(1))
            if before and _variable_y(before, chained=False) == YV:
                return y
        if cue in _Y_DESPEJA:
            after = _NEXT_WORD.match(s, m.end())
            if after and after[1].lower() in _Y_SECOND_VERB:
                return y
        return YV
    if cue == "para" and _Y_SOLVE.search(s, max(0, i - 80), i):
        return YV  # "Resuelve la ecuación para y en..."
    if cue in _Y_VERBS and _Y_MATH_NEXT.match(s, m.end()):
        return YV
    return y


def _operand_before_comma(s: str, i: int) -> bool:
    k = s.rfind(",", 0, i) - 1
    while k >= 0 and s[k] in " \t" + SOFT:
        k -= 1
    if k < 0:
        return False
    if s[k].isascii() and s[k].isalpha():
        return k == 0 or not s[k - 1].isalpha()  # "los ejes x, y"
    return s[k].isdigit() or s[k] in _CLOSERS + YV + ")"


def _is_math(inner: str, *, powered: bool = False) -> bool:
    if not re.search(r"[\da-zA-Zπ]", inner) or _MATH_CONTENT.fullmatch(inner) is None:
        return False
    # Letter runs are variables only when single, a short product behind a coefficient ("2ab"), a short product
    # that is raised to a power ("(ab)^2"), or a function name ("sen" before its argument).
    for run in re.finditer("[a-zA-Z]+", inner):
        word = run[0]
        if len(word) == 1 or (inner[run.end() : run.end() + 1] == FN and word.lower() in _FUNCTIONS):
            continue
        if len(word) <= 3 and (powered or (run.start() > 0 and inner[run.start() - 1].isdigit())):
            continue
        return False
    return True


def _mark_groups(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        text, start, end = m.string, m.start(), m.end()
        bars = m[0][0] == "|"
        inner = next(g for g in m.groups() if g is not None)
        powered = not bars and _next_char(text, end) == "^"
        if not _is_math(inner, powered=powered):
            return _PARK[m[0][0]] + inner + _PARK[m[0][-1]]
        prefix = ABS if bars else ""
        if not bars and _prev_char(text, start) == "^":
            return re.sub(r"\s+", "", inner) if _INT.fullmatch(inner) else LE + inner + RE
        before = text[start - 1] if start else ""
        if not bars and (before == FN or (before in ("f", "g", "h") and not text[start - 2 : start - 1].isalpha())):
            return LF + inner + RF
        core = inner.strip()
        if _NESTED.fullmatch(core) and core[0] in LQ + LS and core[-1] in RQ + RS:
            # "((x+1))" carries one quantity, not two: never "la cantidad la cantidad".
            return LQ + core[1:-1] + RQ if powered else inner
        if powered:
            # "(-3)^2" is "la cantidad menos tres, al cuadrado", never "-3^2" ("menos tres al cuadrado").
            return LQ + inner + RQ
        top, nested = inner, 1
        while nested:
            top, nested = _NESTED.subn("1", top)
        if any(c in _RELATIONS for c in top):
            return SOFT + inner + SOFT  # "(x = 3)" is a statement, not a quantity
        if _TERM.fullmatch(top) or not _BINARY.search(top):
            return prefix + LS + inner + RS
        return prefix + LQ + inner + RQ

    for _ in range(_MAX_DEPTH):
        s, replaced = _GROUP.subn(repl, s)
        if not replaced:
            break
    return s.translate(_UNPARK)


def _fraction(m: re.Match[str]) -> str:
    if len(m[1]) > _MAX_DIGITS or len(m[2]) > _MAX_DIGITS:
        return _chunk(f"{_spell(m[1])} sobre {_spell(m[2])}")
    num, den = int(m[1]), int(m[2])
    nxt = _NEXT_WORD.match(m.string, m.end())
    if num <= den and (
        _SCORE_BEFORE.search(m.string, max(0, m.start() - 48), m.start()) or (nxt and nxt[1].lower() in _SCORE_AFTER)
    ):
        return _chunk(f"{_spell(m[1])} de {_spell(m[2])}")  # "Sacaste 7/10" -> "siete de diez"
    gender = _gender_after(m.string, m.end(), ranged=False)
    if gender is None or num == 0:
        return _chunk(fraction_to_words_es(num, den))
    fem = gender == "f"
    if den == 2:
        # "media taza", "medio kilo", "tres medias tazas"
        half = "media" if fem else "medio"
        return _chunk(half if num == 1 else f"{number_to_words_es(num, fem=fem)} {half}s")
    if nxt[1].lower() in ("parte", "partes") and den in _ORDINALS and den > 2:
        # "2/3 partes" -> "dos terceras partes"
        part = _ORDINALS[den][1]
        return _chunk(f"{number_to_words_es(num, fem=True)} {part if num == 1 else part + 's'}")
    if _denominator(den) is None:
        return _chunk(fraction_to_words_es(num, den))
    return _chunk(fraction_to_words_es(num, den) + " de")  # "tres cuartos de taza"


def _mixed(m: re.Match[str]) -> str:
    if max(len(m[1]), len(m[2]), len(m[3])) > _MAX_DIGITS:
        return m[0]
    whole, num, den = int(m[1]), int(m[2]), int(m[3])
    if not 0 < num < den or _denominator(den) is None:
        return m[0]
    gender = _gender_after(m.string, m.end(), ranged=False)
    if gender is not None and (num, den) == (1, 2):
        fem = gender == "f"
        return _chunk(f"{number_to_words_es(whole, fem=fem)} y {'media' if fem else 'medio'}")  # "dos y media tazas"
    return _chunk(f"{number_to_words_es(whole)} y {fraction_to_words_es(num, den)}")


def _exponent(m: re.Match[str]) -> str:
    # "(x+1)^2" reads "la cantidad x más uno, al cuadrado": the comma keeps the square off the last term.
    lead = m[1] + ", " if m[1] else ""
    if m[3] is not None:
        if len(m[3]) > _MAX_DIGITS:
            words = "a la " + ("menos " if m[2] else "") + _digits(m[3])
        else:
            n = int(m[3])
            if m[2]:
                words = "a la menos " + number_to_words_es(n)
            elif n in (2, 3):
                words = "al cuadrado" if n == 2 else "al cubo"
            else:
                words = "a la " + _POWER_ORDINALS.get(n, number_to_words_es(n))
    elif m[5] is not None:
        words = "a la " + ("menos " if m[4] else "") + m[5]
    elif m[6]:
        return lead + SOFT + "a la potencia" + SOFT + LE
    else:
        words = "a la"
    return lead + _chunk(words)


def _ordinal(m: re.Match[str]) -> str:
    masc, fem, short, suffixes = _ORDINALS[int(m[1])]
    mark = m[2]
    if mark in ("º", "ª"):
        if mark == "ª":
            return _chunk(fem)
        nxt = _NEXT_WORD.match(m.string, m.end())
        return _chunk(short if nxt and _noun_gender(nxt[1]) == "m" else masc)
    if mark not in suffixes:
        return m[0]
    return _chunk(short if mark == "er" else fem if mark.endswith("a") else masc)


def _ordinal_degree(m: re.Match[str]) -> str:
    n = int(m[1])
    masc, _, short, _ = _ORDINALS[n]
    if _MONTH_AFTER.fullmatch(m[2]):
        return _chunk(masc if n == 1 else number_to_words_es(n))  # "el 1° de mayo" -> "el primero de mayo"
    return _chunk(masc if m[2].lower().startswith("de") else short)


def _degrees(m: re.Match[str]) -> str:
    unit = "un grado" if m[1] == "1" else _spell(m[1], apocope=True) + " grados"
    scale = {"C": " Celsius", "F": " Fahrenheit"}.get(m[2] or "", "")
    return _chunk(unit + scale)


def _ends_operand(s: str, i: int) -> bool:
    k = i - 1
    while k >= 0 and s[k] in " \t" + SOFT:
        k -= 1
    if k < 0:
        return False
    c = s[k]
    if c.isascii() and c.isalpha():
        return k == 0 or not s[k - 1].isalpha()
    return c.isdigit() or c in ")" + _CLOSERS + YV


def _starts_operand(s: str, j: int, *, skip: bool) -> bool:
    k = j
    while skip and k < len(s) and s[k] in " \t" + SOFT:
        k += 1
    if k < len(s) and s[k] == "-":
        # "2 * -3", "5 - -3": a sign in front of an operand still starts that operand.
        k += 1
        while k < len(s) and s[k] in " \t" + SOFT:
            k += 1
        if k < len(s) and s[k] == "-":
            return False
    if k >= len(s):
        return False
    c = s[k]
    if c.isascii() and c.isalpha():
        return k + 1 >= len(s) or not s[k + 1].isalpha()
    return c.isdigit() or c in "(" + LQ + LS + OS + YV


def _operator(m: re.Match[str]) -> str:
    op, s, i, j = m[0], m.string, m.start(), m.end()
    if op == "-":
        # A hyphen only reads "menos" next to an operand: keeps "auto-evaluación" and "- paso" bullets as prose.
        if not _starts_operand(s, j, skip=True):
            return op
        if _compound(s, i, j):
            return SOFT  # "COVID-19" is a name with a number, not a subtraction
        prev = _prev_char(s, i)
        after_operator = bool(prev) and prev in _OPS + _OPENERS + "("
        if not (_ends_operand(s, i) or _starts_operand(s, j, skip=False) or after_operator):
            return op
        words = "menos"
    elif op in ("*", "·"):
        # Outside an a*b product an asterisk is markdown emphasis: drop it rather than read "por".
        if not (_ends_operand(s, i) and _starts_operand(s, j, skip=True)):
            return ""
        words = "por"
    elif op == "/":
        words = "sobre" if _ends_operand(s, i) or _starts_operand(s, j, skip=True) else "o"
    else:
        words = _OPERATOR_WORDS[op]
    return SOFT + words + SOFT


def _compound(s: str, i: int, j: int) -> bool:
    if not (j < len(s) and s[j].isdigit() and i > 0 and s[i - 1].isalpha()):
        return False
    k = i
    while k > 0 and s[k - 1].isalpha():
        k -= 1
    word = s[k:i]
    return len(word) >= 4 or (len(word) >= 2 and any(c.isupper() for c in word))


def _noun_gender(word: str) -> str | None:
    if word[0].isupper():
        return None
    lower = word.lower()
    if lower in _FEM_NOUNS or lower.endswith(_FEM_SUFFIXES):
        return "f"
    return "m" if lower in _MASC_NOUNS else None


def _gender_after(s: str, j: int, *, ranged: bool = True, start: int = -1) -> str | None:
    before = _PREV_WORD.search(s, max(0, start - 8), start) if start > 0 else None
    if before and before[1].lower() in ("el", "al", "del"):
        return None  # "El 1 solo se divide", "Al 21 suma 4": the number itself is the noun
    nxt = _NEXT_WORD.match(s, j)
    if nxt is None:
        span = _RANGE_NEXT.match(s, j) if ranged else None
        noun = span and (span[1] or span[2])
        return "f" if noun and _noun_gender(noun) == "f" else None  # "1 a 2 tazas" -> "una a dos tazas"
    word = nxt[1].lower()
    if word in _SOLO:
        noun = _NEXT_WORD.match(s, nxt.end())
        return "m" if noun and _noun_gender(noun[1]) == "m" else None
    gender = _noun_gender(nxt[1])
    if gender is None or _NOUN_IS_VERB.match(s, nxt.end()):
        return None
    if word in _VERB_NOUNS:
        obj = _NEXT_WORD.match(s, nxt.end())
        if obj and obj[1].lower() in _VERB_OBJECT:
            return None  # "La figura 1 muestra que..."
    return gender


def _number(m: re.Match[str]) -> str:
    gender = _gender_after(m.string, m.end(), start=m.start()) if m[2] is None else None
    words = _spell(m[0], fem=gender == "f", apocope=gender == "m")
    if gender and words.endswith(("millón", "millones")):
        words += " de"  # "un millón de personas"
    return SOFT + words + SOFT


def _finish(s: str) -> str:
    s = _CONTRACT.sub(lambda m: m[1][0] + ("l" if len(m[1]) == 1 else "el") + m[2], s)
    s = _NESTED_QUANTITY.sub(LQ + ", ", s)  # "{[2 + 3] × 4}": "la cantidad, la cantidad dos más tres..."
    s = (
        s.replace(EL, "el")
        .replace(YV, "ye")
        .replace(LQ, SOFT + "la cantidad" + SOFT)
        .replace(LF, SOFT + "de" + SOFT)
        .replace(TIMES, SOFT + "por" + SOFT)
    )
    s = re.sub("[\ue000-\ue016]", SOFT, s)

    def collapse(m: re.Match[str]) -> str:
        if " " in m[0] or "\t" in m[0]:
            return " "
        before = m.string[m.start() - 1] if m.start() else ""
        after = m.string[m.end()] if m.end() < len(m.string) else ""
        return " " if before.isalnum() and after.isalnum() else ""

    return _SOFT_RUN.sub(collapse, s)
