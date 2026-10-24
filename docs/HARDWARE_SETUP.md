# Room and hardware setup

How to set up the projector, webcam, whiteboard and lights so the assistant can read what you write,
small symbols included. Budget about 15 minutes the first time.

## Checklist

- [ ] Webcam placed so the reflection of the projector lens falls **off** the projected image (section 1). Projector on a cart in the middle → webcam on a tall tripod **above** the lens (section 0 table), checked with the one-eye test.
- [ ] Matte (anti-glare) whiteboard, clean, no ghost writing.
- [ ] Even room light, no sunlight or spotlight on the board.
- [ ] Black or dark blue marker, not too thin.
- [ ] Camera at 1920×1080 or more (`SAR_CAMERA_WIDTH` / `SAR_CAMERA_HEIGHT`).
- [ ] Projector page full screen, then **Board → Line up camera and projector** on the dashboard.
- [ ] No bright white blob on the board in the camera picture.

## 0. This classroom's kit: Logitech C930e + Epson WXGA 3LCD

**Settings file.** Copy `sar.env.example` to `sar.env` (already filled in for this kit). The app and the
hardware check load it automatically:

| Setting | Value | Why |
| --- | --- | --- |
| `SAR_CAMERA_WIDTH` / `SAR_CAMERA_HEIGHT` | `1920` / `1080` | The C930e's full HD; the high-res ink crop needs it to keep decimals and minus signs sharp |
| `SAR_CAMERA_INDEX` | usually `1` | `0` is normally the laptop's own camera; the hardware check lists the camera names |
| `SAR_PROJECTOR_WIDTH` / `SAR_PROJECTOR_HEIGHT` | `1280` / `800` | WXGA native resolution (16:10). A wrong size skews calibration and pointing |

**First run with the devices plugged in:**

```bash
cp sar.env.example sar.env
.venv/bin/python scripts/hardware_check.py
```

Look for `camera N  PASS  1920x1080 @ ~30 fps` (if it says 1280x720, the C930e is on a USB hub or a
different index) and `projector  PASS  a display runs at 1280x800`. Fix `SAR_CAMERA_INDEX` in `sar.env` if
the 1080p camera is not index 1. If the projector line is missing, or a `displays` line says `skipped`, check
the projector by eye in the display settings (Windows: *Settings → System → Display*; macOS: *System Settings →
Displays*): a second display at 1280 × 800.

**Epson WXGA 3LCD projector**

- Connect it as an **extended** display (not mirrored) at **1280 × 800**: on a Mac, System Settings →
  Displays → the projector → *Use as: Extended display*, resolution 1280 × 800.
- In the projector menu: aspect **Normal/Native** (no stretching); **keystone 0 and Quick Corner off**
  (digital keystone resamples the picture; the camera calibration already corrects the geometry);
  **Auto Iris / dynamic contrast off** (brightness pumping confuses the camera); power or light mode
  **ECO** (less glare). Focus until the calibration checkerboard is crisp.
- 3LCD has no colour wheel, so there is no rainbow flicker in the camera picture.
- **Our setup: projector on a cart, lens level with the middle of the board, square-on.** No keystone is
  needed (leave it at 0). Because the lens is in the middle, its reflection lands in the **middle of the
  image** for any camera near the projector. Put the webcam **above** the lens instead (details below).

**Where the webcam goes with the projector on a cart in the middle.** Mount the C930e on a tall tripod
or light stand, about 0.7 × the image width from the board, and high enough above the lens that the
reflection moves past the top edge of the image: **at least ≈ 0.53 × the image width above the lens**.
Tilt it down (about 35°) so the whole image is in the picture.

| Projected image width | Webcam distance from the board | Webcam height above the lens | Lens at 1.3 m → webcam at |
| --- | --- | --- | --- |
| 1.2 m | ≈ 0.85 m | ≥ 0.65 m | ≈ 1.95 m |
| 1.5 m | ≈ 1.05 m | ≥ 0.80 m | ≈ 2.1 m |
| 2.0 m | ≈ 1.4 m | ≥ 1.05 m | ≈ 2.35 m |

These assume the Epson's usual throw (projector ≈ 1.3–1.6 × the image width from the board) and include a
10% margin. Why it works: the reflection sits between the webcam and the lens (section 1), about two
thirds of the way towards the webcam at these distances, so lifting the webcam lifts the reflection off
the top of the image. At this height the webcam is also outside the projector's light cone (no shadow on
the board), and students writing at the board block it less than a low tripod would. Avoid putting the
webcam beside the projector on the cart: that puts the reflection in the middle of the board.
Finish with the one-eye test (section 1) at the webcam's position.

**Logitech C930e webcam**

- Plug it **directly** into the laptop, not through a hub (hubs often limit it to 720p or 15 fps).
- Open the privacy shutter. Leave the digital zoom at 1× (zoom throws away real pixels).
- In Logitech's camera app (Logi Tune, or Logitech Camera Settings on Windows): **lock focus** (autofocus
  hunts when the projected picture changes), turn **RightLight off**, lock exposure if offered, and set
  **anti-flicker to 60 Hz** (Americas).
- Mount it on a tripod (the clip has a 1/4" thread) so it never moves after calibration.

**How far to put the camera.** The C930e sees about 82° across and 52° up-down. The projected image is
16:10, so its height is what limits you: put the camera about **0.7 × the image width** from the board,
square-on to it.

| Projected image width | Camera distance from the board | Board detail the camera resolves |
| --- | --- | --- |
| 1.2 m | ≈ 0.85 m | ≈ 1.3 px per mm |
| 1.5 m | ≈ 1.05 m | ≈ 1.05 px per mm |
| 2.0 m | ≈ 1.4 m | ≈ 0.8 px per mm |

At about 1 pixel per millimetre, write numbers **at least 5 cm tall**, make decimal points a solid dot
**about 5 mm wide** (the size of a pea), and draw minus signs **as long as a digit is wide**. Exponents can be
smaller, but not less than half the height of the number they belong to.

## 1. Projector and webcam placement

**Why it matters.** A whiteboard is a weak mirror. The webcam can see a reflection of the projector
lens in it: a bright white blob (a *hotspot*). Inside the hotspot the camera is blinded, so marker ink
there disappears and the assistant cannot read it.

**Where the hotspot appears.** Treat the board as a mirror: the lens has a mirror image the same
distance *behind* the board as the lens is in front of it. The camera sees the hotspot where its line of
sight to that mirror image crosses the board. That point is always **between the camera and the lens**,
pulled towards whichever of the two is closer to the board:

```
 side view (looking along the board)

   lens' *            mirror image of the lens (as far behind the board as the lens is in front)
          \
 ==========\===================  whiteboard
            \  <- hotspot: where the camera's line to lens' crosses the board
             \
              [webcam]
                         [projector lens]
```

For a camera at height *c* and distance *d_c* from the board, and a lens at height *p* and distance
*d_p*, the hotspot is at height **(c·d_p + p·d_c) / (d_c + d_p)**; the same formula works sideways. With
equal distances it is simply the midpoint between camera and lens. You only see it where the projector
shows something bright there (the avatar, a speech bubble, the calibration checkerboard), but that is
exactly where the assistant needs to read the board.

**So: put the camera on the same side as the lens, and further out.** Then the midpoint lands beyond the
edge of the projected image:

1. **Projector on a table** (lens below the image): put the webcam **lower than the lens**, e.g. a low
   tripod or the table edge closer to the board, tilted up. The hotspot then falls below the image.
2. **Projector on the ceiling** (lens above the image): put the webcam **higher than the lens**, or on
   the ceiling next to it, tilted down.
3. **Webcam close to the wall, above the board**, looking down at it. Because the camera is much closer
   to the board than the lens, the hotspot sits near the camera, above the image.
4. **Sideways only if it is far enough.** Moving the webcam to the side by an angle of 20–30° only moves
   the hotspot *towards* the edge (by half the sideways offset at equal distances); it stays on the image.
   To push it off, the camera must be more than **about one image width** to the side of the lens at a
   similar distance (image 1.5 m wide → camera more than 1.5 m sideways).
5. **Short-throw projectors** (lens a few cm from the wall, below or above the board): the hotspot sits
   next to the lens, at or beyond the image edge, so a webcam in front of the board is usually fine.

Avoid the opposite side: projector on a table with the webcam high up, or a ceiling projector with the
webcam low, puts the hotspot **in the middle** of the board.

**The one-eye test (always do it).** Open the calibration checkerboard full screen on the projector
(`http://127.0.0.1:5050/calibration.png`, then F11) so the whole image is bright. Stand where the webcam
will go and look at the board with one eye at the webcam's height. If you can see a bright reflection of
the projector lens anywhere on the image, so will the camera: move the webcam further out on the lens's
side until it is gone (with the projector on a cart in the middle, that means higher above the lens, see
section 0). The dashboard's glare check confirms it once the camera is running.

Also:

- The camera must see the **whole projected area** plus a small margin. Fill as much of the picture
  with the board as you can; wasted picture is wasted detail.
- Keep the camera and projector **steady**. Tape the tripod legs. Any bump means calibrating again.
- Do not point the camera at a window or a lamp.

## 2. Whiteboard

- Use a **matte, anti-glare or "projection" whiteboard**. Glossy boards make the hotspot much bigger
  and brighter. A matte whiteboard film or a large flip-chart pad on the board also works.
- **Clean it** with whiteboard cleaner, not just a dry eraser. Ghost writing can be read as ink.
- Keep it **flat**. A curved or bowed board throws off the line-up between camera and projector.

## 3. Lighting

- Room lights **on, but even**. The camera needs light to see the ink: the projector shows pure black
  everywhere except the avatar, so it adds almost no light.
- No **sunlight** or **spotlight** across the board. Close the blinds; switch off a light right above
  the board if it makes a bright patch.
- Too dark: the camera picture looks grainy and faint lines get lost. Too bright: the board looks pure
  white with no texture and the ink looks faded. Aim for a light-grey board with crisp, dark ink in the
  dashboard's camera picture.
- If the projector has an **eco or low-brightness mode**, use it. A dimmer bulb makes a smaller hotspot.

## 4. Markers and writing small symbols

At 1920×1080 the camera sees the board at roughly 1 pixel per millimetre (1920 pixels across a 1.8 m
board). A 2 mm dot is only 2 pixels and is easily lost. Write for the camera:

- Use **black or dark blue**. Red, green and light colours have much less contrast.
- Use a **fresh, full-size marker** (bullet or chisel tip). Fine-tip markers draw lines that are too
  thin to read. A dried-out marker leaves grey, broken lines.
- Write symbols **at least 5 cm tall**.
- **Decimal points:** press the marker down to make a solid dot as big as the tip, with a small gap
  each side. Write a leading zero: `0.25`, not `.25`.
- **Minus signs:** a clear straight dash about as wide as a digit, with a gap before the number so it
  is not read as part of it. Keep it level, so it does not look like an underline.
- **Exponents:** at least half the height of the main number and clearly raised. If space is tight,
  write `10^-3`.
- **Fractions:** a long, straight fraction bar with a gap above and below.
- Keep look-alikes apart: cross your sevens, use `·` for multiply instead of `x`, and leave space
  between steps and boxes.

## 5. Camera resolution

The server opens the camera at 1280×720 by default. That is fine for flowcharts but too coarse for
decimals, minus signs and exponents. Ask for 1920×1080 (or more, if your camera supports it):

macOS / Linux:

```bash
SAR_CAMERA_WIDTH=1920 SAR_CAMERA_HEIGHT=1080 .venv/bin/python -m biobench_sar
```

Windows PowerShell:

```powershell
$env:SAR_CAMERA_WIDTH = "1920"; $env:SAR_CAMERA_HEIGHT = "1080"
.venv\Scripts\python -m biobench_sar
```

Check what you actually get: many webcams quietly fall back to 720p. Open
`http://localhost:5050/api/frame.jpg`, save the picture and check its size. If it is still 1280×720,
plug the camera straight into a USB 3 port (no hub) or try another camera. Then check the **Camera**
frame rate under **More tools**; it should stay around 25–30 fps. If it drops a lot, your camera
cannot keep up at that size; go back to 1280×720 and write bigger.

Make sure the picture is **sharp**: clean the lens, and if the camera's focus keeps hunting, set it
to manual focus in the camera's own app.

## 6. Lining up the camera and projector (calibration)

The assistant has to know which camera pixel lands on which projector pixel, so the avatar can walk to
and point at your drawings. Do this once, and again whenever the camera, projector or board moves, or
you change the projector's zoom or keystone.

1. Start the server.
2. On the screen that goes to the projector, open `http://localhost:5050/` and press `F` for full
   screen. **Do this first**: the checkerboard is drawn on this page, and a page in a window will line
   up wrongly. Move the mouse pointer off the board.
3. On your laptop, open `http://localhost:5050/dashboard` and go to the **Board** tab.
4. Check that the camera picture shows the whole projected area.
5. Step out of the camera's view and click **Line up camera and projector**. A black-and-white
   checkerboard flashes on the board for about two seconds.
6. Read the message under the button. If it says the checkerboard was not detected, see
   [Troubleshooting](#8-troubleshooting).
7. Check it: draw a box on the board. An orange box should appear around it in the dashboard's camera
   picture, in the right place. Under **More tools → Camera view**, **Lined-up** should show the board
   straight and filling the picture.

The result is saved and reused the next time the server starts.

## 7. Checking for glare

- **Eye test:** the one-eye test from section 1, at the webcam's position.
- **Camera picture:** on the **Board** tab (or **More tools → Camera view → Raw camera**), look for a
  bright white blob with no detail. Draw a line through it: if the line fades or vanishes in the
  picture, that spot is glare.
- **Glare warning:** the Board tab's checklist warns you when glare washes out part of the board
  (*A small bright spot on the board*, or *Glare on the board: move the camera*). A small hotspot is a
  heads-up. If more than about 1 % of the board is washed out, the warning is marked serious: fix it
  before you present. The check looks for the bulb's flat white reflection, so Pip and the coloured
  highlight don't trigger it.

To get rid of glare, in order of how much it helps:

1. Move the webcam. With **this kit's cart** (lens level with the middle of the board), **raise the webcam
   higher above the lens** (section 0 table). With a table or ceiling projector, move it further out on the
   lens's side (section 1).
2. Switch the projector to eco mode.
3. Dim the room lights near the board.
4. Use a matte board.

Calibrate again after moving the webcam (section 6).

## 8. Troubleshooting

| What you see | Likely cause | What to do |
| --- | --- | --- |
| Bright white blob in the camera picture; writing there is ignored | Projector bulb reflecting into the camera (glare) | Projector on a cart in the middle (this kit): **raise the webcam higher above the lens** (section 0). Otherwise move it further out on the **same side as the projector lens** (section 1: lower than a table projector, higher than a ceiling one, or more than one image width sideways). Then eco mode, matte board, and calibrate again |
| "Checkerboard not detected" | Projector page not full screen; camera cannot see the whole board; glare over part of the checkerboard; someone in the way | Press `F` on the projector page, re-aim the camera, fix glare, step aside, try again |
| Avatar points next to the drawing, not at it | Camera, projector or board moved since calibration; projector page not full screen | Run **Line up camera and projector** again with the projector page full screen (`F`) |
| The glowing outline is around the wrong part of the board | Same as above: the outline uses the same line-up | Calibrate again |
| No `projector` line, or `displays ... skipped`, in the hardware check | This computer's display list couldn't be read | Check the display settings by eye: a second display at 1280 × 800 |
| `tesseract FAIL` in the hardware check | Tesseract isn't installed (only photo uploads need it) | `winget install UB-Mannheim.TesseractOCR` (Windows) or `brew install tesseract` (macOS). If it is installed and still fails on Windows, add `C:\Program Files\Tesseract-OCR` to your user `PATH` and open a new PowerShell window |
| Decimals, minus signs or exponents misread | Camera at 720p; thin or light marker; symbols too small | 1920×1080 camera; black marker; bigger, clearer symbols (section 4) |
| No orange boxes around drawings | Faint, dry or light-coloured marker; uneven light | Fresh black or blue marker; even lighting |
| Board looks pure white and the ink looks faded everywhere | Camera over-exposed | Dim the lights near the board; lower exposure in the camera's own app |
| Picture dark and grainy | Room too dark | Turn some room lights on |
| Camera stays at 1280×720 | Camera does not support 1080p or the USB link is too slow | USB 3 port without a hub, or another camera |
| Camera frame rate drops a lot | Camera cannot send that size fast enough | Use 1280×720 and write bigger |
| Old, erased writing still detected | Ghosting on the board | Clean with whiteboard cleaner |
| Picture blurry | Camera out of focus | Clean the lens; set manual focus |
