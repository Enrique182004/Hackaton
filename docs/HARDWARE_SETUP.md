# Room and hardware setup

How to set up the projector, webcam, whiteboard and lights so the assistant can read what you write,
small symbols included. Budget about 15 minutes the first time.

## Checklist

- [ ] Webcam **not** next to the projector lens: 20–30° to the side, or above or below it.
- [ ] Matte (anti-glare) whiteboard, clean, no ghost writing.
- [ ] Even room light, no sunlight or spotlight on the board.
- [ ] Black or dark blue marker, not too thin.
- [ ] Camera at 1920×1080 or more (`SAR_CAMERA_WIDTH` / `SAR_CAMERA_HEIGHT`).
- [ ] Projector page full screen, then **Board → Line up camera and projector** on the dashboard.
- [ ] No bright white blob on the board in the camera picture.

## 1. Projector and webcam placement

**Why it matters.** A whiteboard is a weak mirror. Light from the projector bounces off it at the same
angle it arrived, like a ball off a wall. If the webcam sits on that bounce path it sees the projector's
bulb as a bright white blob (a *hotspot*). Inside the hotspot the camera is blinded: marker ink there
disappears, so the assistant cannot read it, and drawings there go undetected.

The bounce goes back on the mirror side of the board's straight-out line. With the projector straight in
front of the board, the bounce comes right back at the projector, so a webcam sitting next to the lens
gets the full glare. Move the webcam off that line:

```
                 whiteboard
   ==================================
                    |  \
                    |    \
                    |      \   25°
                    |        \
                    |          \
               [projector]    [webcam]

   The projector's light hits the board and bounces straight back down the
   vertical line, into anything sitting beside the lens. The webcam, 20–30°
   off that line, is out of the bounce.
```

Good set-ups, best first:

1. **Webcam to one side.** Tripod 20–30° to the side of the projector's line, pointing back at the
   middle of the board. Rough sideways distances:

   | Camera distance from the board | Move the camera sideways by |
   | --- | --- |
   | 1.5 m | 55–85 cm |
   | 2 m | 75–115 cm |
   | 3 m | 1.1–1.7 m |

2. **Webcam above or below.** If the projector is on a table, put the webcam high (top of a tripod,
   on a shelf) looking down at the board. If the projector is on the ceiling, put the webcam low.
3. **Short-throw or ceiling projector.** These already hit the board at a steep angle; the bounce goes
   down to the floor or up to the ceiling, so a webcam in front of the board is usually fine.

**The one-eye test.** Before you fix the webcam in place, stand where it will go and look at the board
with one eye at the webcam's height. If you can see a bright reflection of the projector lens, so will
the camera. Move until it is gone.

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
- **Glare warning:** the dashboard's Board tab will warn you when glare washes out part of the board.
  A small hotspot is a heads-up. If more than about 1 % of the board is washed out, the warning is
  marked serious: fix it before you present. (The warning appears once the glare check is switched on
  in the server. Until then, use the two checks above.)

To get rid of glare, in order of how much it helps: move the webcam sideways (or up or down), switch
the projector to eco mode, dim the room lights near the board, use a matte board.

## 8. Troubleshooting

| What you see | Likely cause | What to do |
| --- | --- | --- |
| Bright white blob in the camera picture; writing there is ignored | Projector bulb reflecting into the camera (glare) | Move the webcam 20–30° sideways or above/below the projector; eco mode; matte board |
| "Checkerboard not detected" | Projector page not full screen; camera cannot see the whole board; glare over part of the checkerboard; someone in the way | Press `F` on the projector page, re-aim the camera, fix glare, step aside, try again |
| Avatar points next to the drawing, not at it | Camera, projector or board moved since calibration | Run **Line up camera and projector** again |
| Decimals, minus signs or exponents misread | Camera at 720p; thin or light marker; symbols too small | 1920×1080 camera; black marker; bigger, clearer symbols (section 4) |
| No orange boxes around drawings | Faint, dry or light-coloured marker; uneven light | Fresh black or blue marker; even lighting |
| Board looks pure white and the ink looks faded everywhere | Camera over-exposed | Dim the lights near the board; lower exposure in the camera's own app |
| Picture dark and grainy | Room too dark | Turn some room lights on |
| Camera stays at 1280×720 | Camera does not support 1080p or the USB link is too slow | USB 3 port without a hub, or another camera |
| Camera frame rate drops a lot | Camera cannot send that size fast enough | Use 1280×720 and write bigger |
| Old, erased writing still detected | Ghosting on the board | Clean with whiteboard cleaner |
| Picture blurry | Camera out of focus | Clean the lens; set manual focus |
