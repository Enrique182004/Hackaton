# Phones and tablets: the remote dashboard

The assistant needs the laptop: the camera, the models, the microphone and the projector all live there. A phone
or tablet can't run it, but on the same Wi-Fi it can open the laptop's dashboard and drive it remotely. The laptop
still does all the work.

**Status: groundwork only.** The pieces are in the repo but the server doesn't use them yet, so `SAR_LAN=1`
does nothing until someone completes the [wiring spec](#wiring-spec-for-the-integrator) below. The pieces are:

| File | What it is |
| --- | --- |
| `biobench_sar/lan.py` | This machine's LAN addresses, `join_url`, the `JoinPin` PIN and cookie gate, and the host and loopback checks |
| `biobench_sar/static/manifest.webmanifest`, `static/icons/` | Add to Home Screen: name, colours, icons (regenerate them with `scripts/make_icons.py`) |
| `biobench_sar/static/sw.js` | A small service worker that makes reloads faster (see its limits below) |
| `biobench_sar/static/vendor/qrcode.min.js` | qrcode-generator 1.4.4 (MIT), unmodified from cdnjs. Its licence is in `qrcode.LICENSE.txt` |

## What works on a phone

- **Asking questions by typing.** To ask by voice, use the keyboard's **dictation microphone** (the mic key on
  the iPhone keyboard, or voice typing in Gboard). The phone turns the speech into text, so the browser never
  needs microphone permission.
- **Seeing the answers** and the assistant's status as they happen, through the same Socket.IO updates the laptop
  dashboard gets.
- **Changing the mode** (step-by-step help, similar problems or quiz) and **starting or stopping a quiz**.
- **Choosing the study files**: tick or untick worksheets and notes. Adding a file from the phone uses the same
  picker; on most phones it can take a photo of a worksheet.
- **Add to Home Screen** gives the phone an icon that opens the dashboard full screen.

## What doesn't work, and the workarounds

- **The phone's microphone inside the page.** Browsers only allow `getUserMedia` on HTTPS or `localhost`, and
  the laptop serves plain HTTP on the Wi-Fi. Use keyboard dictation instead. The dashboard's listen button on a
  phone starts the **laptop's** microphone, which is useful when the student is at the board.
- **Spoken answers** play in the room through the laptop and projector, not on the phone.
- **Using the phone as the camera.** This isn't possible through the browser. As an optional workaround, make the
  phone a webcam for the laptop, then point `SAR_CAMERA_INDEX` at it and check it with
  `scripts/hardware_check.py` (it needs at least 28 FPS):
  - **Continuity Camera** (iPhone with iOS 16+ and a Mac with macOS 13+): the iPhone appears as an ordinary camera.
  - **DroidCam** (Android or iPhone with Windows, macOS or Linux): an app on the phone plus a virtual-webcam
    client on the laptop.
- **Offline use and a true installed app on the phone.** Service workers and installable web apps need HTTPS too,
  so over plain-HTTP Wi-Fi the phone gets a home-screen shortcut without offline caching. The service worker only
  runs on the laptop (`http://127.0.0.1` counts as secure).
- **iPhone home-screen icon.** It keeps its own cookies, separate from Safari's, so the first launch from the icon
  shows the PIN form once. Type the PIN.

## How a teacher turns it on (after the wiring lands)

1. **Same network.** The laptop and the phones join the same Wi-Fi. Guest and campus networks often use
   *client isolation*, which stops devices from reaching each other. If the phone can't load the page, use a
  phone hotspot or a router of your own.
2. **Start the assistant with `SAR_LAN=1`** (or add `SAR_LAN=1` to `sar.env`):

   ```bash
   SAR_LAN=1 .venv/bin/python -m biobench_sar
   ```

   ```powershell
   $env:SAR_LAN='1'; .\.venv\Scripts\python.exe -m biobench_sar
   ```

   The first time, the system asks whether Python may accept incoming connections. On Windows, allow
   **Private networks**; on macOS, click **Allow**.
3. **On the laptop**, open `http://127.0.0.1:5050/dashboard` as usual. A **"Use a phone or tablet"** card shows
   a QR code and a six-digit PIN. The terminal prints the same address and PIN.
4. **On the phone**, scan the QR code with the camera app. It opens the dashboard and the phone is in. Without the
   QR code, type the address shown under it (`http://<laptop-ip>:5050/join`) and then the PIN.
5. A device that has joined stays in for 12 hours or until the assistant restarts. **Every restart makes a new
   PIN**, and the card's "New PIN" button makes one at any time.

## Security model

- **Off by default.** Without `SAR_LAN=1` the server listens on `127.0.0.1` only, so nothing on the network can
  reach it.
- **With `SAR_LAN=1`**, the server listens on `0.0.0.0`, so anyone on the same network can reach it. Without a
  valid cookie they get only the join form and the static files (CSS, JS, icons). Everything else needs the
  cookie: pages, `/api/*`, `/tts/*`, camera frames and Socket.IO.
- **The PIN** has six digits drawn with `secrets` and is compared in constant time (`hmac.compare_digest`).
  After **10 wrong tries within a minute**, the join form locks for **60 seconds** for everyone. At that rate,
  guessing a PIN takes weeks. The PIN lives in memory only, so every restart makes a new one, and
  `JoinPin.rotate()` backs the "New PIN" button.
- **The cookie** (`sar_lan`) carries a 256-bit random token. The server keeps only its SHA-256 hash, in memory,
  for 12 hours. The cookie is `HttpOnly`, `SameSite=Lax` and `Path=/`, and not `Secure` because the LAN is plain
  HTTP. Restarting the assistant signs every device out.
- **The laptop itself never needs the PIN.** Requests from a loopback address (`127.0.0.1`, `::1`) pass. Open the
  dashboard on the laptop as `http://127.0.0.1:5050`. Through its own LAN address the laptop counts as just
  another device.
- **DNS rebinding is blocked.** The `Host` header must be an IP address, `localhost` or a name you configure.
  Without this check, a web page that points its own domain at `127.0.0.1` could borrow the laptop's loopback
  trust.
- The existing same-origin guards stay: `block_cross_origin` for POSTs, and Socket.IO's
  `cors_allowed_origins=None`.
- **Known limits.** On an open Wi-Fi network (no password), anyone can sniff the plain HTTP traffic, PIN and
  cookie included. WPA2/WPA3 networks encrypt each device's traffic, so use one of those, never public Wi-Fi.
  A device with the cookie has the dashboard's full powers: files, modes and quizzes. And because the lockout
  is shared, a student who spams wrong PINs can make everyone else wait a minute. The teacher can press
  New PIN.

## Wiring spec (for the integrator)

Everything below goes into files other agents own: `config.py`, `app.py`, `templates/*` and
`static/js/dashboard/*`. Each step names the `lan.py` function to use. Ship tests with each step (see the end).

1. **Settings (`config.py`).** Add `lan: bool` (true when `SAR_LAN=1`). **When `SAR_LAN=1`, `SAR_HOST` becomes
   `0.0.0.0`**, meaning `host` defaults to `"0.0.0.0" if lan else "127.0.0.1"`. An explicit `SAR_HOST` still wins;
   if it is a loopback address while `lan` is on, log a warning, because phones can't reach it.

2. **One gate per server (`create_app`).** When `settings.lan` is on, create `gate = JoinPin()` and store it in
   `app.config["SAR_JOIN_PIN"]`. Also call `mimetypes.add_type("application/manifest+json", ".webmanifest")`,
   because Windows' MIME table may not have it.

3. **`before_request` gate**, registered after `block_cross_origin`:

   ```python
   from biobench_sar.lan import COOKIE_NAME, is_allowed_host, is_local_request, is_request_allowed


   @app.before_request
   def lan_gate() -> Any:
       if not settings.lan:
           return None
       if not is_allowed_host(request.host):
           return jsonify({"error": "unknown host"}), 400  # refused before /join, so this can't loop
       if is_request_allowed(
           gate,
           remote_addr=request.remote_addr,
           host=request.host,
           path=request.path,
           token=request.cookies.get(COOKIE_NAME),
       ):
           return None
       if request.path.startswith("/api/") or request.method != "GET":
           return jsonify({"error": "join with the PIN first"}), 401
       return redirect("/join")
   ```

   Pass `request.remote_addr`, the TCP peer. Never trust `X-Forwarded-For`: there is no proxy.

4. **`GET /join`** shows the PIN form and sets the cookie on success. `POST /join` (form field `pin`) handles a
   typed PIN the same way.
   - A local request (`is_local_request`) redirects to `/dashboard` at once, since the laptop never needs a PIN.
   - With `?pin=` (from the QR code) or a POSTed `pin`, call `token = gate.redeem(pin)`. On success, call
     `response.set_cookie(COOKIE_NAME, token, **cookie_kwargs())` and send a **303 to `/dashboard`**, which
     takes the PIN out of the address bar and the history.
   - On failure, show the form again with "That PIN didn't work". While `gate.locked_for() > 0`, show
     "Too many wrong PINs. Try again in N seconds."
   - Send `Cache-Control: no-store` and `Referrer-Policy: no-referrer` on every `/join` response.
   - Add a new `templates/join.html`: a light page in the dashboard's style. Give the input
     `inputmode="numeric" autocomplete="one-time-code" maxlength="9"`; `verify()` ignores spaces and dashes.

5. **Socket.IO `connect` is gated the same way.** Flask's `before_request` never runs for `/socket.io/`, because
   python-socketio answers before Flask. At the top of `on_connect`, add:

   ```python
   if settings.lan and not is_request_allowed(
       gate, remote_addr=request.remote_addr, host=request.host, path="/socket.io/", token=request.cookies.get(COOKIE_NAME)
   ):
       raise ConnectionRefusedError("join with the PIN first")
   ```

   With `always_connect=True`, the CONNECT is acknowledged before this runs, and `async_handlers` runs events on
   their own threads. So repeat the same check at the top of `text_query` and `trigger_listen`. `flask.request`
   there is the connection's handshake request, cookies included.

6. **Dashboard card "Use a phone or tablet"**, shown only when LAN mode is on and the viewer is the laptop.
   - **Local-only endpoints.** `GET /api/lan` returns `{"enabled": true, "pin": gate.pin, "locked_for": …,
     "urls": [join_url(ip, settings.port, gate.pin) for ip in local_ip_addresses()]}`. `POST /api/lan/pin`
     calls `gate.rotate()`. Both answer 403 unless `is_local_request(request.remote_addr)`, so a joined phone
     never sees the PIN. With LAN mode off, `GET /api/lan` returns `{"enabled": false}`.
   - **The card.** Show the QR of `urls[0]`, the PIN in large type grouped as `123 456`, the address
     (`urls[0]` without the `?pin=`) for typing, any other addresses in small type, and a "New PIN" button.
     If `urls` is empty, say "This laptop isn't on a network. Join the classroom Wi-Fi."
   - **QR code.** Load `{{ url_for('static', filename='vendor/qrcode.min.js') }}` as a classic script (it
     defines a global `qrcode`), then render:

     ```js
     const qr = qrcode(0, "M");  // 0 = smallest version that fits
     qr.addData(url);
     qr.make();
     box.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 4, scalable: true });
     ```

7. **Startup log (`main`).** When LAN mode is on, print the laptop's address as `http://127.0.0.1:<port>`
   (`0.0.0.0` can't be clicked). Then print `Phones  <join_url without ?pin>  PIN 123 456` for each address and
   a line about the firewall prompt.

8. **Templates.** Add these to `precharge.html` and `join.html`. The projector page doesn't need them.

   ```html
   <link rel="manifest" href="{{ url_for('static', filename='manifest.webmanifest') }}">
   <meta name="theme-color" content="#23497d">
   <link rel="icon" href="{{ url_for('static', filename='icons/icon.svg') }}" type="image/svg+xml">
   <link rel="apple-touch-icon" href="{{ url_for('static', filename='icons/apple-touch-icon.png') }}">
   <meta name="apple-mobile-web-app-title" content="Whiteboard">
   ```

9. **Service worker.** Serve it from the root, because a worker served at `/static/sw.js` can't control
   `/dashboard`:

   ```python
   @app.get("/sw.js")
   def service_worker() -> Any:
       response = send_from_directory(app.static_folder, "sw.js", mimetype="text/javascript", max_age=0)
       response.headers["Cache-Control"] = "no-cache"
       return response
   ```

   Register it with `if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});`.
   `serviceWorker` is missing on plain-HTTP phones, so they skip it.
   - **Register it only when LAN mode is on.** Its stale-while-revalidate strategy serves the *previous* JS and
     CSS on the first reload after an edit, which is confusing during development.
   - When the cached shell list changes, bump `CACHE_VERSION` in `sw.js`.
   - What it caches:
     - Page loads are network first, with the cached page after 4 seconds or when the laptop is unreachable.
     - `/static/*` is stale-while-revalidate.
     - `/api/*` is network first. A stored JSON read is served only when the network fails, marked with
       `X-SAR-From-Cache: 1`.
     - It never touches Socket.IO, non-GET requests, `/join`, camera frames, `/api/listen`, `/tts/*`, other
       origins or redirects (a redirect is the gate sending a device to `/join`).

10. **Tests to ship with the wiring.**
    - **LAN mode off:** nothing changes.
    - **LAN mode on:**
      - Use `client.get(..., environ_base={"REMOTE_ADDR": "192.168.1.30"})` for a phone.
      - `/dashboard` redirects to `/join`, and `/api/status` returns 401.
      - `/join?pin=<right>` sets `sar_lan` and returns 303 to `/dashboard`; with the cookie, `/dashboard` returns 200.
      - A wrong PIN 10 times locks the form.
      - `Host: evil.example` returns 400.
      - Loopback passes without a cookie.
      - `/api/lan` returns 403 from a phone.
      - For Socket.IO, use `socketio.test_client(app, flask_test_client=client)` so the cookie carries over; a
        refused connect leaves `is_connected()` false.
