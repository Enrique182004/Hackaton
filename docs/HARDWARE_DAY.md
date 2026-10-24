# Hardware day: plug in and test

One ordered run-through for the first session with the real kit: a **Logitech C930e** webcam and an
**Epson WXGA 3LCD** projector on a cart, with its lens level with the middle of the whiteboard. Each step
says what you should see. If you don't see it, use the fix in that step before moving on.

Budget about 45 minutes. Do part A the day before.

---

## A. The day before (no hardware needed)

1. **Install.** Either the desktop build (Windows installer or macOS app, see
   [DESKTOP_BUILDS.md](DESKTOP_BUILDS.md)) or the source checkout (`bash scripts/setup_mac.sh --pull-model`,
   Windows: `scripts/setup_windows.ps1`).
2. **The AI model.** Install Ollama and run `ollama pull qwen2.5vl:3b` (about 3.2 GB, needs internet once).
   ✅ `ollama list` shows `qwen2.5vl:3b`.
3. **Room settings.** Copy `sar.env.example` to `sar.env`: next to the app on a desktop install, or in
   the repo folder on a source checkout. It is already set for this kit: camera 1920×1080, projector
   1280×800, math tutor, language Auto.
4. **Dry run.** Start the assistant (`.venv/bin/python -m biobench_sar`, or open the app) and open
   `http://127.0.0.1:5050/dashboard`.
   ✅ The dashboard opens. **Study files** shows the three seed files (quiz bank, tutoring guide, worksheet)
   and **Ask** answers a typed question within ~10 s ("Help me with the next step").
   Board shows "Camera not found". That is expected without the webcam.
5. **First-run download.** On a computer with a microphone the speech model (~140 MB) downloads in the
   background at the first start. Let it finish while you still have internet.

## B. Set up the room

6. **Projector on the cart**, lens level with the middle of the board and square-on to it (no tilt).
   Projector menu: aspect **Normal/Native**, **keystone 0**, **Quick Corner off**, **Auto Iris off**,
   **ECO** light mode. Connect it to the laptop as an **extended display** at **1280 × 800**.
   ✅ The laptop's display settings show a second display at 1280 × 800.
7. **Webcam above the lens.** The C930e goes on a tall tripod about **0.7 × the image width** from the
   board and at least **0.53 × the image width above the lens**, tilted down so the whole projected
   image is in view. For a 1.5 m image: about 1.05 m from the board, at least 0.8 m above the lens. Table
   for other sizes: [HARDWARE_SETUP.md, section 0](HARDWARE_SETUP.md).
   Plug it **directly** into the laptop (no hub). In Logitech's camera app: **lock focus**, **RightLight
   off**, **anti-flicker 60 Hz**. Open the privacy shutter.
8. **Board and markers.** Clean, matte board. Black or dark-blue marker. Write numbers at least 4 cm tall,
   decimal points like a pea, minus signs as long as a digit is wide.

## C. Check the hardware

9. **Run the hardware check** (from the repo folder; the desktop app shows the same information on its
   dashboard):

   ```bash
   .venv/bin/python scripts/hardware_check.py
   ```

   ✅ Expect: `settings file PASS`, `camera names` listing **Logitech Webcam C930e**, one
   `camera N PASS 1920x1080 @ ~30 fps`, `microphone PASS`, `tesseract PASS`, `ollama PASS`,
   `model qwen2.5vl:3b PASS`, `projector PASS a display runs at 1280x800`.
   - Camera gives 1280x720 → it is on a hub or another camera; plug it directly.
   - The 1080p camera is not index 1 → set `SAR_CAMERA_INDEX` in `sar.env` to the index that showed
     1920x1080, then restart the assistant.
   - Microphone FAIL on a Mac → allow the microphone for Terminal (or the app) in System Settings →
     Privacy & Security → Microphone; same for Camera.

## D. Line up camera and projector

10. **Projector page.** Open `http://127.0.0.1:5050/` on the projector display, click once (allows
    sound), then press **F** for full screen.
    ✅ The board shows Pip, the avatar, on a black (invisible) background.
11. **Calibrate.** On the laptop dashboard: **Board** tab → **Line up camera and projector**. A checkerboard
    flashes on the board for a few seconds; keep everything still.
    ✅ "Lined up!" and the Board checklist shows **Camera and projector are lined up**.
    - "Couldn't see the checkerboard" → the projector page is not full screen, the camera does not see the
      whole image, or the room is too bright. Fix and press it again.
12. **Glare.** The Board checklist shows no glare warning.
    ✅ "No glare on the board".
    - Glare warning → the projector lens is reflecting into the camera. Move the webcam further out on the
      lens's side (higher above it), then use the one-eye test from
      [HARDWARE_SETUP.md, section 1](HARDWARE_SETUP.md), and calibrate again.
13. **Ink.** Write `3x + 5 = 20` large on the board.
    ✅ Board tab: an orange box appears around it within a second or two (More tools → "Drawings: 1").

## E. First lessons (English)

Ask from the dashboard (**Ask** tab: type, or press **Talk**) or press **Space** on the projector page.

| Say / type | Expected |
| --- | --- |
| "Help me with the next step" (with `3x + 5 = 20` on the board) | Pip points at the equation, the equation is highlighted, and the next step is explained ("subtract five from both sides"). **It never says x = 5.** |
| "Give me a problem like this one" | A new two-step equation is dictated in words; it's shown in notation on the dashboard. |
| "Quiz me" | A quiz starts. Problem 1 of 10 is read out and shown in a card on the projector. Write the answer, then say "I'm done". A right answer gets 🎉 and the next problem; a wrong one gets an amber hint, never the answer. |
| "I want to learn how to solve fractions" (empty board) | A short intro, then: "Do you have an exercise in mind, or should I make one up for you?" Answer "make one up" → a starter problem is dictated. |
| "Stop the quiz" | Pip asks to confirm; say "yes". |

✅ The voice never reads symbols ("three x plus five equals twenty", not "3x+5=20").

## F. First lessons (Spanish)

Switch the dashboard language to **Español** (or leave **Auto** and just speak Spanish).

| Di / escribe | Esperado |
| --- | --- |
| "Ayúdame con el siguiente paso" | Explica el siguiente paso en español, sin decir el resultado. |
| "Dame otro problema parecido" | Dicta una ecuación nueva en palabras ("dos x más tres es igual a once"). |
| "Examíname" | Empieza el quiz en español. |
| "Quiero aprender a resolver fracciones" | Una introducción corta y la pregunta «¿Tienes un ejercicio en mente o quieres que invente uno?»; responde «invéntame uno». |

✅ The Spanish voice is a Mexican voice and reads math in Spanish words.

## G. Write down what you saw

Record for each step: ✅ or what happened instead, with a phone photo of the board and the dashboard.
Attach the `hardware_check.py` output. These close GitHub issues **#2** (camera and mic), **#3**
(calibration on a real board), **#7** (wake word, if you try `SAR_LISTEN_MODE=wake`) and **#9** (scenario
test cases). File anything that went wrong as a new issue with the photos.

## Quick fixes

| Symptom | Fix |
| --- | --- |
| Dashboard says it can't reach the assistant | The assistant window was closed. Start it again. |
| Answers say the AI model is unreachable | Start Ollama (open the Ollama app), then ask again. |
| Avatar points next to the writing | Calibrate again (step 11); check the projector page is full screen. |
| Small symbols misread (decimals, minus signs) | Write bigger and bolder; check the camera really runs at 1920×1080 (step 9). |
| Washed-out spot on the board in the camera view | Glare: step 12. |
| No sound | Click once on the projector page; check the laptop's output device. |
