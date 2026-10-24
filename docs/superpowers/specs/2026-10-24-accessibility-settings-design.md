# Accessibility settings — design

Date: 2026-10-24 · Branch: `feat/accessibility-settings` from `release/candidate` (one PR).

## Goal

A teacher can make the assistant easier to use for a student with a disability, from an **Accessibility** card
under **More tools** on the dashboard (laptop or a joined phone). The settings apply to the whole room: the
projector, the voice and every open dashboard follow them at once, and they survive a restart.

| Setting | Values | Who it helps |
| --- | --- | --- |
| Captions stay up | on / off (default off) | Deaf and hard-of-hearing students, slower readers |
| Text size on the board | normal · large (1.25×) · extra large (1.5×) | Low-vision students, students at the back |
| Voice speed | slower (0.75×) · normal · faster (1.25×) | Processing differences, hard of hearing; faster for screen-reader users |
| Calm mode | on / off (default off) | Autism, ADHD, motion or light sensitivity |
| High contrast | on / off (default off) | Low vision, colour-blind students |

## Settings model (`biobench_sar/accessibility.py`, new — contract M)

- `AccessibilitySettings` (frozen dataclass): `captions_stay: bool = False`, `text_size: str = "normal"`,
  `speech_rate: str = "normal"`, `calm: bool = False`, `high_contrast: bool = False`.
  - `TEXT_SIZES = ("normal", "large", "xlarge")`, `SPEECH_RATES = {"slower": 0.75, "normal": 1.0, "faster": 1.25}`.
  - `to_dict()` returns all five keys; `rate_factor` returns the float for `speech_rate`.
- `parse_changes(body: object) -> dict[str, object]`: accepts a JSON object holding any non-empty subset of the five
  keys. Booleans must be real JSON booleans; strings must match exactly (case-sensitive, like `/api/language`).
  Anything else (not an object, empty, unknown key, wrong type or value) raises `ValueError` with an English
  message naming the key and its allowed values.
- `AccessibilityStore(path)`: thread-safe; loads once in `__init__`; a missing, unreadable or invalid file (or
  any invalid key in it) gives the defaults for those keys. `current` returns the settings; `update(changes)`
  applies already-parsed changes, saves, and returns the new settings. A failed save logs a warning and keeps the
  in-memory value (the same rule as the document selection).
- File: `Settings.accessibility_file` — env `SAR_ACCESSIBILITY_FILE`, default `DATA_DIR / "accessibility.json"`,
  read inside the `default_factory` lambda (the desktop launcher patches `DATA_DIR`).

### One atomic JSON write (`biobench_sar/jsonfile.py`, new)

`save_json(path, payload, what) -> bool`: `mkdir(parents)`, `mkstemp` in the same folder, `json.dump`,
`os.replace`; on `OSError` logs `could not save {what} to {path}: {exc}`, removes the temp file and returns
`False`. `DocumentSelection._save` and `QuizBank._save` switch to it (they are identical copies today). The
store uses it too. `Calibration.save` raises on failure on purpose and keeps its own code (listed in the PR's
**Cleanup** section).

## Server (`assistant.py`, `app.py`)

- `Assistant` owns an `AccessibilityStore(settings.accessibility_file)`.
  - `accessibility_body() -> dict` — the five keys.
  - `set_accessibility(changes) -> dict` — `parse_changes`, `store.update`, push the rate to the voice, emit
    `accessibility` with the full body to every client, return the body.
  - The voice gets the saved rate when the services are attached, and on every change, through a
    `getattr(tts, "set_rate", None)` + `callable` guard (the same guard `set_language` uses for audio), so test
    fakes without `set_rate` keep working.
  - `estimate_speech_seconds(text, rate=1.0)` divides the words-per-second part by `rate`; `_run` passes the
    current factor, so slower speech keeps the wake-word loop deaf for longer. The default stays `2.6 s` for the
    existing test.
- Routes (inside `create_app`, behind the existing LAN PIN gate — no `only_the_laptop()`: phones may change them):
  - `GET /api/accessibility` → the five keys.
  - `PUT /api/accessibility` with a partial body → the full body, or `400 {"error": …}` and nothing saved or
    pushed.
- Socket: `on_connect` sends `accessibility` to the new client after `language`. No client-to-server event.
- `page()` passes the current settings to the projector template so the first paint is already right.

## Voice speed (`llm/tts_engine.py`)

`TTSEngine.set_rate(factor: float)` stores the factor (clamped to 0.5–2.0). At `1.0` nothing extra is passed
anywhere, so today's commands and tests are unchanged.

- edge-tts: `Communicate(text, voice, rate=f"{round((factor - 1) * 100):+d}%")` only when the factor is not 1.
- macOS `say`: adds `-r <round(175 × factor)>` (175 words per minute is the system default rate).
- pyttsx3: the queue item carries the factor; the worker reads the engine's base `rate` once and sets
  `base × factor` when it is not 1.

## Projector (`templates/index.html`, `static/js/main.js`, `static/css/style.css`)

The settings live on `<body>` as `data-text-size="normal|large|xlarge"`, `data-captions="stay|timed"`,
`data-calm="on|off"`, `data-contrast="on|off"`, rendered by Flask and then kept current by the `accessibility`
event and by `GET /api/accessibility` in `resync()` (with a version counter, like the language). After any
change: `fitBubble(); uiChanged();` so placement and the masked `ui_rects` follow at once.

- **Captions stay up** (`data-captions="stay"`): `armBubbleTimer` never arms a hide timer. The bubble is replaced
  by the next answer and cleared as soon as a question starts (status `listening`, `transcribing` or `thinking`,
  like the quiz card; the typing box opening; or the `transcript` event), so
  it never covers the board while the camera reads it. The student's heard words (`#transcript`) stay until the
  answer arrives instead of 5 s. Switching captions off while a caption is held arms a 4 s hide.
- **Text size**: a `--text-scale` (1 / 1.25 / 1.5) chosen by `body[data-text-size]` multiplies the font sizes of
  the bubble, the quiz card label and problem, the transcript and the status label. The bubble's max width grows
  with it, capped at `50vw`. The quiz card's `max-width` stays at most `30vw` (its camera test).
- **Calm mode** (`data-calm="on"`): the same rules as the existing `prefers-reduced-motion` block (no bob, hop,
  shake, nudge, highlight pulse, status ring or dots), plus: the avatar glides on `left`/`top` only (no sprite
  slide), and the glows (`drop-shadow`, `text-shadow`, `box-shadow`) are softened. The calibration checkerboard
  is not touched. The OS reduced-motion setting keeps working as before.
- **High contrast** (`data-contrast="on"`): bubble border 6 px, pure white text, no blurred text glow; the error
  bubble's border is dashed and the celebrating bubble's is double, so correct and wrong differ by more than
  colour. The highlight outline is 6 px, and the JS geometry (`HIGHLIGHT_BORDER`) follows it. The quiz card
  text and label go to full opacity; the card's border stays at most 2 px with no glow (the camera reads thick
  bright lines as ink). The background stays `#000000`.

## Dashboard (`templates/precharge.html`, `static/js/dashboard/accessibility.js` new, `static/css/dashboard.css`, `static/js/i18n.js`)

- An **Accessibility** card (`h3`) at the top of the third **More tools** column, above **Settings**, so a LAN
  laptop with the phone card still fits one screen.
  - Three checkboxes (`.check-line`): keep captions up until the next question; calm mode (no bouncing or
    flashing); high contrast.
  - Two segmented groups: text size on the board (Normal · Large · Extra large) and voice speed (Slower · Normal ·
    Faster). Buttons carry ids and `data-access-*` attributes and `aria-pressed`.
  - A one-line hint ("These change the projector and the voice for the whole room.") and a `.note[role=status]`
    for failures.
- `accessibility.js` exports `initAccessibility`, `refreshAccessibility` and `renderAccessibility`:
  - The card stays `hidden` until the first `GET /api/accessibility` succeeds (an older server leaves it hidden).
  - A change disables the card's controls, sends `PUT /api/accessibility` with just that key, renders the reply,
    and re-enables in `finally`. A failure leaves the controls as they were, shows a translated note (`warn`)
    and logs the raw reason in the Activity log.
  - It follows `accessibility` socket pushes and refreshes on every socket `connect`; without a live socket the
    2 s status poll refreshes it. A version counter stops a late GET from undoing a newer push.
- `camera.js` scopes its handler to its own group (`#camera-view`) instead of every `.seg` on the page.
- New `access.*` keys in English and Mexican Spanish (tú form). `Normal` is the same in both languages.

## Docs

- `docs/BUILD_PLAN.md`: contract **M — `accessibility.py`**, the route and event rows next to `/api/language`,
  the on-connect list, the Projector UI paragraph, a Shared `accessibility_file` line, a Status row and an
  Implementation-notes row (accessibility is additive to both specs).
- `README.md`: `SAR_ACCESSIBILITY_FILE` under **Files**; the dashboard section names **More tools →
  Accessibility**; a short "Accessibility" paragraph near "English and Spanish".
- `CONTRIBUTING.md`: the new modules in the modules table.
- `docs/MOBILE.md`: phones can change the Accessibility card.

## Tests

- `tests/test_accessibility.py` (new): defaults; `parse_changes` accepts each key and rejects non-objects, empty
  bodies, unknown keys, wrong types and wrong values; the store saves and reloads; a broken file or bad values
  give defaults; a failed save keeps the value; `rate_factor`.
- `tests/test_jsonfile.py` (new): `save_json` writes atomically, reports failure, and the selection and quiz bank
  save through it.
- `tests/test_app.py`: GET/PUT round trip, partial updates, 400 leaves the file and clients untouched, every page
  hears the push, the on-connect list gains `accessibility`, the projector template gets the body attributes.
- `tests/test_lan_wiring.py`: a phone without the PIN gets 401 on `PUT /api/accessibility`; the route is
  `no-store`.
- `tests/test_tts.py`: rate strings for edge, `-r` for `say`, the pyttsx3 `rate` property; nothing changes at 1.0.
- `tests/test_assistant.py`: the speaking estimate scales with the rate; the voice gets the rate on attach and on
  change; fakes without `set_rate` still work.
- `tests/test_frontend.py`: projector body attributes follow the event and the GET; captions stay (no hide timer,
  cleared on `transcript`/`thinking`, 4 s hide when switched off); the dashboard card PUTs one key, follows a
  push, shows a note on failure and stays hidden on 404; EN/ES parity and template checks pass;
  `DEFAULT_ROUTES` gains `GET /api/accessibility`.

## Out of scope

Named per-student profiles; dashboard text size (browser zoom already works, sizes are in rem); simpler wording
from the model; "read my board to me"; voice-only solving; braille export; a dyslexia font on the projector.
