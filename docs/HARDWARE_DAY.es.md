# Día del equipo: conectar y probar

*In English: [HARDWARE_DAY.md](HARDWARE_DAY.md).*

Esta guía recorre, en orden, la primera sesión con el equipo del salón: una cámara web **Logitech C930e** y
un proyector **Epson WXGA 3LCD** sobre un carrito frente a un pizarrón de **1.40 m × 1.11 m**. Cada
paso dice qué debes ver. Si no lo ves, aplica la solución de ese paso antes de seguir.

Calcula unos 45 minutos y haz la parte A un día antes.

---

## A. Un día antes (sin el equipo)

1. **Instalación.** Usa la versión de escritorio (instalador de Windows o app de macOS, ver
   [DESKTOP_BUILDS.md](DESKTOP_BUILDS.md), en inglés) o el código fuente
   (`bash scripts/setup_mac.sh --pull-model`, o `scripts/setup_windows.ps1` en Windows).
2. **El modelo de IA.** Instala Ollama y ejecuta `ollama pull qwen2.5vl:3b` (unos 3.2 GB; necesita internet
   una sola vez).
   ✅ `ollama list` muestra `qwen2.5vl:3b`.
3. **Ajustes del salón.** Copia `sar.env.example` como `sar.env`: junto a la app si es la versión de
   escritorio, o en la carpeta del repositorio si es el código fuente (`cp sar.env.example sar.env`; en
   PowerShell: `Copy-Item sar.env.example sar.env`). Ya viene listo para este equipo: cámara de 1920×1080 en el
   índice 1, proyector de 1280×800, tutor de matemáticas e idioma automático.
4. **Prueba en seco.** Inicia el asistente (`.venv/bin/python -m biobench_sar`, o abre la app) y abre
   `http://127.0.0.1:5050/dashboard`.
   ✅ Se abre el panel. **Ejercicios y apuntes** muestra los tres archivos de ejemplo (banco de quiz, guía
   de tutoría y hoja de ejercicios), y **Preguntar** contesta una pregunta escrita en unos 10 s («Ayúdame con
   el siguiente paso»). La pestaña **Pizarrón** dice «No se encontró la cámara», y eso es normal sin la cámara
   web.
5. **Descarga del primer arranque.** En una computadora con micrófono, el modelo de voz (~140 MB) se
   descarga en segundo plano la primera vez que arranca. Déjalo terminar mientras tengas internet.
6. **Voz en español sin internet (opcional).** La voz mexicana es en línea (Edge). Sin internet, el
   asistente usa una voz instalada en la computadora, así que instala una de español (México): en macOS,
   Ajustes del Sistema → Accesibilidad → Contenido leído → Voz del sistema → Administrar voces (por ejemplo,
   *Paulina*); en Windows, Configuración → Hora e idioma → Idioma y región → agrega *Español (México)* con
   texto a voz. Con `SAR_TTS_BACKEND=say` (Mac) o `pyttsx3` (Windows) en `sar.env` ni siquiera intenta la voz
   en línea. Pruébala: apaga el Wi-Fi, haz una pregunta en español y escucha.

## B. Preparar el salón

7. **Proyector en el carrito**, de frente al pizarrón y nivelado. En el menú del proyector: aspecto
   **Normal/Nativo**, **corrección trapezoidal (keystone) en 0**, **Quick Corner desactivado**, **iris
   automático desactivado** y modo de luz **ECO**. Conéctalo a la laptop como **pantalla extendida** a
   **1280 × 800**. Para ver la imagen completa, abre `http://127.0.0.1:5050/calibration.png` en la pantalla
   del proyector, en pantalla completa (F11; en una Mac, Ctrl+Cmd+F).
   Nuestro pizarrón mide **1.40 m × 1.11 m**: ajusta el zoom para que la imagen mida unos **1.35 m de ancho**
   (0.84 m de alto), con unos 13 cm de pizarrón libre arriba y abajo. Luego sube o baja el carrito hasta que
   la imagen quede centrada. En la mayoría de los proyectores Epson la imagen sale hacia arriba desde la
   lente, así que la lente queda a unos **20 cm arriba del borde de abajo del pizarrón**, no a media altura.
   Si el carrito no baja tanto, inclina el frente del proyector unos 10° hacia abajo con su pata (el keystone
   se queda en 0: la calibración corrige el pequeño trapecio).
   ✅ La configuración de pantallas de la laptop muestra una segunda pantalla de 1280 × 800, y la imagen cabe
   en el pizarrón.
8. **La cámara web**, centrada, a **0.95 m del pizarrón**. Su altura depende de dónde quedó la lente:
   - Lente cerca de la parte de abajo de la imagen (proyector nivelado): cámara unos **10 cm abajo del borde
     de abajo del pizarrón**, en un tripié bajo, inclinada hacia arriba para que vea toda la imagen.
   - Lente a media altura (proyector inclinado): cámara al menos **0.75 m arriba de la lente**, en un tripié
     alto, inclinada hacia abajo.

   Nunca la pongas justo al lado de la lente: su reflejo caería en medio del pizarrón. Detalles:
   [HARDWARE_SETUP.md, sección 0](HARDWARE_SETUP.md#0-this-classrooms-kit-logitech-c930e--epson-wxga-3lcd)
   (en inglés). Conéctala **directo** a la laptop, sin hub. En la app de cámara de Logitech: **bloquea el
   enfoque**, **apaga RightLight** y pon el **antiparpadeo en 60 Hz**. Abre la tapa de privacidad.
9. **Pizarrón y plumones.** Pizarrón limpio y mate, plumón negro o azul oscuro. Escribe los números de al
   menos 5 cm de alto, el punto decimal del tamaño de un chícharo y el signo menos tan largo como el ancho de
   un dígito.

## C. Revisar el equipo

10. **Corre la revisión del equipo** desde la carpeta del repositorio:

    ```bash
    .venv/bin/python scripts/hardware_check.py
    ```

    (En PowerShell: `.\.venv\Scripts\python.exe scripts\hardware_check.py`. La app de escritorio no trae esta
    revisión: la lista de la pestaña **Pizarrón** de su panel muestra la cámara, el micrófono y el modelo.)

    ✅ Debes ver: `settings file PASS`, `camera names` con **Logitech Webcam C930e**, un
    `camera N PASS 1920x1080 @ ~30 fps`, `microphone PASS`, `tesseract PASS`, `ollama PASS`,
    `model qwen2.5vl:3b PASS` y `projector PASS a display runs at 1280x800`.
    - La cámara da 1280x720 → está en un hub o es otra cámara. Conéctala directo.
    - La cámara de 1080p no es el índice 1 → pon en `SAR_CAMERA_INDEX` de `sar.env` el índice que mostró
      1920x1080 y reinicia el asistente.
    - `microphone FAIL` en una Mac → permite el micrófono para Terminal (o para la app) en Ajustes del
      Sistema → Privacidad y seguridad → Micrófono. Haz lo mismo con la Cámara.
    - Cámara o micrófono sin imagen ni sonido en Windows → Configuración → Privacidad y seguridad → Cámara (y
      Micrófono) → activa **Permitir que las aplicaciones de escritorio accedan a la cámara**.
    - No sale la línea `projector`, o `displays ... skipped` → revisa a ojo en la configuración de pantallas
      (Windows: Configuración → Sistema → Pantalla; macOS: Ajustes del Sistema → Pantallas) que el proyector
      esté a 1280 × 800.
    - `tesseract FAIL` → instálalo (`winget install UB-Mannheim.TesseractOCR` en Windows, `brew install
      tesseract` en Mac). Solo hace falta para subir fotos. Si en Windows sigue fallando después de
      instalarlo, agrega `C:\Program Files\Tesseract-OCR` al `PATH` de tu usuario y abre otra ventana de
      PowerShell.

## D. Alinear la cámara y el proyector

11. **Página del proyector.** Abre `http://127.0.0.1:5050/` en la pantalla del proyector, haz clic una vez
    (así deja sonar el audio) y presiona **F** para pantalla completa.
    ✅ En el pizarrón aparece Pip, el avatar, sobre un fondo negro que no se ve.
12. **Alinear.** En el panel de la laptop: pestaña **Pizarrón** → **Alinear cámara y proyector**. Un tablero
    de ajedrez aparece unos segundos en el pizarrón, así que no muevas nada.
    ✅ «¡Listo, quedó alineado!», y la lista del Pizarrón muestra **La cámara y el proyector están
    alineados**.
    - «No se vio el tablero de ajedrez» → la página del proyector no está en pantalla completa, la cámara no
      ve toda la imagen o hay demasiada luz en el salón. Corrígelo y vuelve a presionar el botón.
13. **Reflejos.** La lista del Pizarrón no muestra ningún aviso de reflejo.
    ✅ «No hay reflejos en el pizarrón».
    - Aviso de reflejo → la lente del proyector se refleja en la cámara. Aleja más la cámara web de la lente
      hacia arriba o hacia abajo: **más abajo** si está debajo del pizarrón, **más arriba** si está encima de
      la lente. Compruébalo con la prueba de un ojo de
      [HARDWARE_SETUP.md, sección 1](HARDWARE_SETUP.md#1-projector-and-webcam-placement) (en inglés: párate
      donde va la cámara y mira el pizarrón con un solo ojo a su altura; si ves el reflejo de la lente, la
      cámara también lo ve) y vuelve a alinear.
14. **Tinta.** Escribe `3x + 5 = 20` en grande en el pizarrón.
    ✅ En la pestaña Pizarrón aparece un recuadro naranja alrededor en uno o dos segundos (Más herramientas →
    «Dibujos: 1»). Pip, el globo de diálogo y la tarjeta del quiz nunca llevan recuadro: la página del proyector
    le dice al servidor dónde están, y esa parte de la imagen de la cámara se ignora.

## E. Primeras clases (inglés)

Pregunta desde el panel (pestaña **Preguntar**: escribe o presiona **Hablar**) o presiona **Espacio** en la
página del proyector. Las frases van en inglés, tal cual.

| Di / escribe | Esperado |
| --- | --- |
| "Help me with the next step" (con `3x + 5 = 20` en el pizarrón) | Pip camina justo a la izquierda de la ecuación y la señala, y un contorno brillante late alrededor de la ecuación (se desvanece a los 8 s, más o menos). Explica el siguiente paso ("subtract five from both sides"). **Nunca dice x = 5.** |
| "Give me a similar problem" | Dicta en palabras una ecuación nueva de dos pasos, que se ve escrita en el panel. Pip se queda donde no tapa nada de lo escrito. |
| "Quiz me" | Empieza un quiz. Lee el problema 1 de 10 y lo muestra en una tarjeta arriba a la izquierda del proyector. Escribe la respuesta y di "check my answer". El programa califica lo que escribiste: si está bien hay 🎉 y el siguiente problema; si está mal, una pista en ámbar (nunca la respuesta) y el contorno alrededor de tu respuesta. Si no puede leer la respuesta, te dice que no pudo leerla bien y que se la vuelvas a pedir. |
| "I'm stuck" (durante el quiz) | Una pista sobre el mismo problema. El quiz y la calificación no cambian. |
| "I want to learn how to solve fractions" (pizarrón vacío) | Una introducción corta y luego "Do you have an exercise in mind, or should I make one up for you?". Responde "make one up" y dicta un problema para empezar. |
| "Stop the quiz" | Pip pregunta "Do you want to end the quiz? Say yes to stop, or keep going." Di "yes" y te dice tu calificación. Cualquier otra cosa sigue con el quiz. El botón **Terminar quiz** del panel lo termina de inmediato. |

✅ La voz nunca lee símbolos ("three x plus five equals twenty", no "3x+5=20").

## F. Primeras clases (español)

Cambia el idioma del panel a **Español** (o deja **Auto** y simplemente habla en español).

| Di / escribe | Esperado |
| --- | --- |
| «Ayúdame con el siguiente paso» | Explica el siguiente paso en español, sin decir el resultado, y resalta la ecuación. |
| «Dame un problema parecido» | Dicta una ecuación nueva en palabras («dos x más tres es igual a once»). |
| «Hazme un quiz» (o «Examíname») | Empieza el quiz en español. Una respuesta escrita como `x = 5` o «x es igual a cinco» se califica igual. |
| «Quiero aprender a resolver fracciones» | Una introducción corta y la pregunta «¿Tienes un ejercicio en mente o quieres que invente uno?». Responde «invéntame uno». |
| «Termina el quiz» | Pregunta: ¿Quieres terminar el quiz? Di «sí» para terminar o sigue con tu respuesta. Responde «sí» (o «claro», «dale») y te dice tu calificación. |

✅ La voz en español es mexicana y lee las matemáticas con palabras en español.

## G. Celulares (opcional)

Los celulares y tabletas conectados al mismo Wi-Fi pueden servir como panel a distancia, aunque todo el
trabajo lo sigue haciendo la laptop. Detalles y límites: [MOBILE.md](MOBILE.md) (en inglés).

15. **Actívalo.** Detén el asistente, agrega `SAR_LAN=1` a `sar.env` y vuelve a iniciarlo. Acepta el aviso
    del firewall (Windows: **Redes privadas**; macOS: **Permitir**).
    ✅ La terminal (o la ventana de consola de la app) muestra la dirección para los celulares, por ejemplo
    `http://192.168.1.20:5050/join`, y un **PIN** de seis dígitos. En la laptop, que sigue usando
    `http://127.0.0.1:5050/dashboard` y nunca necesita el PIN, **Más herramientas → Usar un celular o una
    tableta** muestra la misma dirección y el PIN con un código QR.
16. **Entra desde un celular.** Conecta el celular al mismo Wi-Fi y escanea el código QR con su cámara, o abre
    la dirección y escribe el PIN.
    ✅ El celular muestra el panel. Una pregunta escrita en el celular (o dictada con el micrófono del
    teclado) hace que Pip conteste en el pizarrón.
    - La página no carga → los dos deben estar en el mismo Wi-Fi, y las redes de invitados o de la escuela
      muchas veces no dejan que los dispositivos se vean entre sí (usa el hotspot de un celular). Desconecta
      cualquier VPN de la laptop.
    - «Demasiados PIN incorrectos» → ese celular espera un minuto. Los que ya entraron siguen funcionando.
17. **Al terminar la clase.** En la misma tarjeta, presiona **Desconectar todos los celulares** (o **PIN
    nuevo**, que también los desconecta a todos), quita `SAR_LAN=1` y reinicia. Cada reinicio crea un PIN
    nuevo.

## H. Anota lo que viste

Marca cada paso con ✅ o escribe qué pasó en su lugar, con una foto del pizarrón y del panel tomada con el
celular. Adjunta la salida de `hardware_check.py`. Con esto se cierran los issues de GitHub **#2** (cámara y
micrófono), **#3** (alineación en un pizarrón real), **#7** (palabra de activación, si pruebas
`SAR_LISTEN_MODE=wake`; en español responde a «asistente») y **#9** (casos de prueba). Lo que haya salido mal,
repórtalo como un issue nuevo con las fotos.

## Soluciones rápidas

| Qué pasa | Qué hacer |
| --- | --- |
| El panel dice que no hay conexión con el asistente | Se cerró la ventana del asistente. Vuelve a iniciarlo. |
| Las respuestas dicen que no se puede llegar al modelo de IA | Inicia Ollama (abre la app de Ollama) y vuelve a preguntar. |
| El avatar señala junto a lo escrito, o el contorno brillante sale en otro lugar | Vuelve a alinear (paso 12) y revisa que la página del proyector esté en pantalla completa. |
| Lee mal los símbolos pequeños (decimales, signos menos) | Escribe más grande y más grueso, y revisa que la cámara de verdad esté a 1920×1080 (paso 10). |
| Dice una y otra vez que no pudo leer tu respuesta | Escribe la respuesta sola en el último renglón, grande, como `x = 5`. |
| Una mancha deslavada en el pizarrón en la imagen de la cámara | Es un reflejo: ve el paso 13. |
| No hay sonido | Haz clic una vez en la página del proyector y revisa la salida de audio de la laptop. |
| El español suena con voz en inglés (sin internet) | Instala una voz de español (México) (paso 6). |
