# Hardware day: plug in and test

*En español: [HARDWARE_DAY.es.md](HARDWARE_DAY.es.md).*

This is one ordered run-through for the first session with the real kit: a **Logitech C930e** webcam and an
**Epson WXGA 3LCD** projector on a cart in front of a **1.40 m × 1.11 m** whiteboard. Each step
says what you should see. If you don't see it, use the fix in that step before moving on.

Budget about 45 minutes, and do part A the day before.

---

## A. The day before (no hardware needed)

1. **Install.** Use either the desktop build (Windows installer or macOS app, see
   [DESKTOP_BUILDS.md](DESKTOP_BUILDS.md)) or the source checkout (`bash scripts/setup_mac.sh --pull-model`,
   or `scripts/setup_windows.ps1` on Windows).
2. **The AI model.** Install Ollama and run `ollama pull qwen2.5vl:3b` (about 3.2 GB, needs internet once).
   ✅ `ollama list` shows `qwen2.5vl:3b`.
3. **Room settings.** Copy `sar.env.example` to `sar.env`: next to the app on a desktop install, or in
   the repo folder on a source checkout (`cp sar.env.example sar.env`; PowerShell:
   `Copy-Item sar.env.example sar.env`). It is already set for this kit: camera 1920×1080 at index 1, projector
   1280×800, math tutor, language Auto.
4. **Dry run.** Start the assistant (`.venv/bin/python -m biobench_sar`, or open the app) and open
   `http://127.0.0.1:5050/dashboard`.
   ✅ The dashboard opens. **Worksheets & notes** lists the three seed files (quiz bank, tutoring guide,
   worksheet), and **Ask** answers a typed question within ~10 s ("Help me with the next step").
   The Board tab shows "Camera not found", which is expected without the webcam.
5. **First-run download.** On a computer with a microphone, the speech model (~140 MB) downloads in the
   background at the first start. Let it finish while you still have internet.
6. **Spanish voice offline (optional).** The Mexican Spanish voice is online (Edge). Without internet the
   assistant falls back to a voice installed on the laptop, so install a Spanish (Mexico) one: on macOS,
   System Settings → Accessibility → Spoken Content → System voice → Manage Voices (e.g. *Paulina*); on
   Windows, Settings → Time & language → Language & region → add *Español (México)* with text-to-speech.
   `SAR_TTS_BACKEND=say` (Mac) or `pyttsx3` (Windows) in `sar.env` skips the online voice altogether.
   Test it: with the Wi-Fi off, ask one question in Spanish and listen.

## B. Set up the room

7. **Projector on the cart**, square-on to the board and level. In the projector menu: aspect
   **Normal/Native**, **keystone 0**, **Quick Corner off**, **Auto Iris off**, **ECO** light mode. Connect it
   to the laptop as an **extended display** at **1280 × 800**. To see the whole picture, open
   `http://127.0.0.1:5050/calibration.png` on the projector display, full screen (F11; on a Mac,
   Ctrl+Cmd+F).
   Our board is **1.40 m × 1.11 m**: zoom so the picture is about **1.35 m wide** (0.84 m tall), with about
   13 cm of board free above and below. Then raise or lower the cart until the picture is centered. With
   most Epson projectors the picture rises from the lens, so the lens ends up about **20 cm above the board's
   bottom edge**, not at mid-board. If the cart can't go that low, tip the projector's front down about 10°
   with its foot (keystone stays 0: calibration handles the slight trapezoid).
   ✅ The laptop's display settings show a second display at 1280 × 800, and the picture fits on the board.
8. **Webcam**, centered, **0.95 m from the board**. Its height depends on where the lens ended up:
   - Lens near the bottom of the picture (projector level): webcam about **10 cm below the board's bottom
     edge** on a short tripod, tilted up so the whole picture is in view.
   - Lens at mid-board (projector tipped down): webcam at least **0.75 m above the lens** on a tall stand,
     tilted down.

   Never put it right beside the lens: its reflection would land in the middle of the board. Details:
   [HARDWARE_SETUP.md, section 0](HARDWARE_SETUP.md#0-this-classrooms-kit-logitech-c930e--epson-wxga-3lcd).
   Plug it **directly** into the laptop (no hub). In Logitech's camera app: **lock focus**, **RightLight
   off**, **anti-flicker 60 Hz**. Open the privacy shutter.
9. **Board and markers.** Use a clean, matte board and a black or dark-blue marker. Write numbers at least
   5 cm tall, decimal points the size of a pea, and minus signs as long as a digit is wide.

## C. Check the hardware

10. **Run the hardware check** from the repo folder:

    ```bash
    .venv/bin/python scripts/hardware_check.py
    ```

    (PowerShell: `.\.venv\Scripts\python.exe scripts\hardware_check.py`. The desktop app has no hardware check:
    its dashboard's Board checklist shows the camera, microphone and model instead.)

    ✅ Expect: `settings file PASS`, `camera names` listing **Logitech Webcam C930e**, one
    `camera N PASS 1920x1080 @ ~30 fps`, `microphone PASS`, `tesseract PASS`, `ollama PASS`,
    `model qwen2.5vl:3b PASS`, `projector PASS a display runs at 1280x800`.
    - Camera gives 1280x720 → it is on a hub, or it is another camera. Plug it in directly.
    - The 1080p camera is not index 1 → set `SAR_CAMERA_INDEX` in `sar.env` to the index that showed
      1920x1080, then restart the assistant.
    - Microphone FAIL on a Mac → allow the microphone for Terminal (or the app) in System Settings →
      Privacy & Security → Microphone. Do the same for Camera.
    - Camera or microphone dark on Windows → *Settings → Privacy & security → Camera* (and *Microphone*) →
      turn on **Let desktop apps access your camera**.
    - No `projector` line, or `displays ... skipped` → check by eye that the display settings show the
      projector at 1280 × 800.
    - `tesseract FAIL` → install it (`winget install UB-Mannheim.TesseractOCR` on Windows, `brew install
      tesseract` on a Mac). Only photo uploads need it. If Windows still fails after installing, add
      `C:\Program Files\Tesseract-OCR` to your user `PATH` and open a new PowerShell window.

## D. Line up camera and projector

11. **Projector page.** Open `http://127.0.0.1:5050/` on the projector display, click once (this allows
    sound), then press **F** for full screen.
    ✅ The board shows Pip, the avatar, on a black (invisible) background.
12. **Calibrate.** On the laptop dashboard, go to the **Board** tab → **Line up camera and projector**. A
    checkerboard flashes on the board for a few seconds, so keep everything still.
    ✅ "Lined up!", and the Board checklist shows **Camera and projector are lined up**.
    - "Couldn't see the checkerboard" → the projector page is not full screen, the camera does not see the
      whole image, or the room is too bright. Fix it and press the button again.
13. **Glare.** The Board checklist shows no glare warning.
    ✅ "No glare on the board".
    - Glare warning → the projector lens is reflecting into the camera. Move the webcam further from the
      lens up-down: **lower** if it sits below the board, **higher** if it sits above the lens. Check with the
      one-eye test from
      [HARDWARE_SETUP.md, section 1](HARDWARE_SETUP.md#1-projector-and-webcam-placement), then calibrate again.
14. **Ink.** Write `3x + 5 = 20` large on the board.
    ✅ On the Board tab, an orange box appears around it within a second or two (More tools → "Drawings: 1").
    Pip, the speech bubble and the quiz card are never boxed: the projector page tells the server where they
    are, and that part of the camera picture is ignored. Neither is the board's metal frame at the left or right
    edge when the picture spills past the board.

## E. First lessons (English)

Ask from the dashboard (**Ask** tab: type, or press **Talk**) or press **Space** on the projector page.

| Say / type | Expected |
| --- | --- |
| "Help me with the next step" (with `3x + 5 = 20` on the board) | Pip walks to just left of the equation and points at it, and a glowing outline pulses around the equation (it fades after about 8 s). The next step is explained ("subtract five from both sides"). **It never says x = 5.** |
| "Give me a similar problem" | A new two-step equation is dictated in words and shown in notation on the dashboard. Pip stands where it covers none of the writing. |
| "Quiz me" | A quiz starts. Problem 1 of 10 is read out and shown in a card at the top left of the projector. Write the answer, then say "check my answer". The code grades what you wrote: a right answer gets 🎉 and the next problem, and a wrong one gets an amber hint (never the answer) with the outline around your answer. An answer it can't read gets "I couldn't read your answer clearly. Ask me to check it again." |
| "I'm stuck" (during the quiz) | One hint on the same problem. The quiz and the score stay as they are. |
| "I want to learn how to solve fractions" (empty board) | A short intro, then "Do you have an exercise in mind, or should I make one up for you?" Answer "make one up" and a starter problem is dictated. |
| "Stop the quiz" | Pip asks "Do you want to end the quiz? Say yes to stop, or keep going." Say "yes" and you hear your score. Anything else keeps the quiz going. The dashboard's **Stop quiz** button stops at once. |

✅ The voice never reads symbols ("three x plus five equals twenty", not "3x+5=20").

## F. First lessons (Spanish)

Switch the dashboard language to **Español** (or leave **Auto** and just speak Spanish).

| Di / escribe | Esperado |
| --- | --- |
| "Ayúdame con el siguiente paso" | Explica el siguiente paso en español, sin decir el resultado, y resalta la ecuación. |
| "Dame un problema parecido" | Dicta una ecuación nueva en palabras ("dos x más tres es igual a once"). |
| "Hazme un quiz" (o "Examíname") | Empieza el quiz en español. Una respuesta escrita como "x = 5" o "x es igual a cinco" se califica igual. |
| "Quiero aprender a resolver fracciones" | Una introducción corta y la pregunta «¿Tienes un ejercicio en mente o quieres que invente uno?»; responde «invéntame uno». |
| "Termina el quiz" | «¿Quieres terminar el quiz? Di «sí» para terminar o sigue con tu respuesta.» Responde «sí». |

✅ The Spanish voice is a Mexican voice and reads math in Spanish words.

## G. Phones (optional)

Phones and tablets on the same Wi-Fi can be remote dashboards. The laptop still does all the work.
Details and limits: [MOBILE.md](MOBILE.md).

15. **Turn it on.** Stop the assistant, add `SAR_LAN=1` to `sar.env` and start it again. Allow the firewall
    prompt (Windows: **Private networks**; macOS: **Allow**).
    ✅ The terminal (or the app's console window) prints the phone address, e.g.
    `http://192.168.1.20:5050/join`, and a six-digit **PIN**. On the laptop, which keeps using
    `http://127.0.0.1:5050/dashboard` and never needs the PIN, **More tools → Use a phone or tablet** shows
    the same address and PIN with a QR code.
16. **Join from a phone.** Put the phone on the same Wi-Fi and scan the QR code with its camera, or open the
    address and type the PIN.
    ✅ The phone shows the dashboard. A question typed on the phone (or spoken with the keyboard's dictation
    mic) makes Pip answer on the board.
    - The page doesn't load → both must be on the same Wi-Fi, and guest or campus networks often block devices
      from reaching each other (use a phone hotspot). Disconnect any VPN on the laptop.
    - "Too many wrong PINs" → that phone waits a minute. Phones that already joined keep working.
17. **End of class.** On the phone card, press **Disconnect all phones** (or **New PIN**, which also signs
    every phone out), then remove `SAR_LAN=1` and restart. A restart always makes a new PIN.

## H. Write down what you saw

Record each step as ✅, or write what happened instead, with a phone photo of the board and the dashboard.
Attach the `hardware_check.py` output. These close GitHub issues **#2** (camera and mic), **#3**
(calibration on a real board), **#7** (wake word, if you try `SAR_LISTEN_MODE=wake`) and **#9** (scenario
test cases). File anything that went wrong as a new issue with the photos.

## Quick fixes

| Symptom | Fix |
| --- | --- |
| Dashboard says it can't reach the assistant | The assistant window was closed. Start it again. |
| Answers say the AI model is unreachable | Start Ollama (open the Ollama app), then ask again. |
| Avatar points next to the writing, or the glowing outline is in the wrong place | Calibrate again (step 12) and check the projector page is full screen. |
| Small symbols misread (decimals, minus signs) | Write bigger and bolder, and check the camera really runs at 1920×1080 (step 10). |
| "I couldn't read your answer clearly" again and again | Write the answer alone on the bottom line, large, as `x = 5`. |
| Washed-out spot on the board in the camera view | Glare: see step 13. |
| No sound | Click once on the projector page and check the laptop's output device. |
| Spanish sounds like an English voice (offline) | Install a Spanish (Mexico) voice (step 6). |
