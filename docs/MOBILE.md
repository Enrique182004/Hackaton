# Phones and tablets: the remote dashboard

The assistant needs the laptop: the camera, the models, the microphone and the projector all live there. A phone
or tablet can't run it, but on the same Wi-Fi it can open the laptop's dashboard and drive it remotely. The laptop
still does all the work.

**Status: groundwork only.** The pieces are in the repo but the server doesn't use them yet, so `SAR_LAN=1`
does nothing until someone completes the [wiring spec](#wiring-spec-for-the-integrator) below. The pieces are:

| File | What it is |
| --- | --- |
| `biobench_sar/lan.py` | This machine's LAN addresses, `join_url`, the `JoinPin` PIN and cookie gate, the host and loopback checks, `SocketIOGate` and `PhoneSessions` |
| `biobench_sar/static/manifest.webmanifest`, `static/icons/` | Add to Home Screen: name, colours, icons (regenerate them with `scripts/make_icons.py`) |
| `biobench_sar/static/sw.js` | A small service worker for LAN mode that makes reloads faster on the laptop (see its limits below) |
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
   **Private networks**; on macOS, click **Allow**. Setting `SAR_HOST` to anything other than `127.0.0.1`,
   `localhost` or `::1` (for example `0.0.0.0`) also turns LAN mode on, PIN included.
3. **On the laptop**, open `http://127.0.0.1:5050/dashboard` as usual. A **"Use a phone or tablet"** card shows
   a QR code and a six-digit PIN. The terminal prints the same address and PIN.
4. **On the phone**, scan the QR code with the camera app. It opens the dashboard and the phone is in. Without the
   QR code, type the address shown under it (`http://<laptop-ip>:5050/join`) and then the PIN.
5. A device that has joined stays in for 12 hours or until the assistant restarts. **Every restart makes a new
   PIN.** On the card:
   - **New PIN** makes a new PIN and signs every phone out; they scan the new QR code to come back. Use it if
     the PIN has leaked.
   - **Disconnect all phones** signs every phone out and closes their live connections, but keeps the PIN. Use it
     at the end of a class.
6. **When class is over**, restart the assistant without `SAR_LAN=1`. A server left in LAN mode overnight gives a
   guesser more time (see the odds below).

### If a phone can't connect

- **It can't load the page at all.** Check that both are on the same Wi-Fi and that the firewall prompt was
  allowed. Guest and campus networks often isolate devices: use a phone hotspot or your own router.
- **Disconnect any VPN on the laptop.** A full-tunnel VPN makes the laptop report its VPN address, which the QR
  code then points to and phones can't reach. Corporate and campus VPNs often do this.
- **Try the other addresses** listed in small type under the QR code, if there are any.
- **"Too many wrong PINs"** means that device has to wait. Joined devices keep working.

## Security model

- **Off by default.** With `SAR_LAN` unset and `SAR_HOST` unset or loopback (`127.0.0.0/8`, `::1`,
  `localhost`), the server listens on this machine only, so nothing on the network can reach it.
- **Every network bind is gated.** LAN mode is on whenever `SAR_LAN=1` **or** `SAR_HOST` is anything else
  (`0.0.0.0`, `::`, an empty string, a LAN address or a host name). There is no way to listen on the network
  without the PIN gate. Anyone on the same network can then reach the server, but without a valid cookie they get
  only the join form and the static files (CSS, JS, icons). Everything else needs the cookie: pages, `/api/*`,
  `/tts/*`, camera frames and Socket.IO.
- **The PIN** has six digits drawn with `secrets` and is compared in constant time (`hmac.compare_digest`).
  It lives in memory only. `/api/*` responses are `no-store`, and the service worker never stores them, so the
  PIN that `GET /api/lan` returns is never written to a cache.
- **Guessing limits.** Wrong PINs count **per device**, keyed on the TCP peer address (for IPv6 the `/64`,
  because one phone can use many addresses):
  - 5 wrong PINs within a minute lock that device for 1 minute. Each further lockout doubles (2, 4, 8, ...
    minutes) up to **1 hour**. A right PIN, or a day without wrong PINs, starts that device over.
  - On top of that, **all devices together** get at most **50 wrong PINs an hour**. Past that, nobody can join
    until the oldest miss is an hour old. Devices that have already joined keep working. One device on its own
    makes at most 30 wrong guesses in an hour, so it can never reach this cap alone.
  - Guesses made while locked out are refused and don't count.
- **The odds, honestly.** A guess is right one time in a million. The table assumes a script guessing as fast as
  the limits allow, the assistant running around the clock in LAN mode, and nobody pressing New PIN:

  | Who is guessing | Guesses a day | Chance per day | Chance over a 5-day school week |
  | --- | --- | --- | --- |
  | One device, one address | about 145 on day 1, then 120 | about 0.015% | about 0.06% |
  | A script that keeps changing its address (the server-wide cap) | 1,200 | about 0.12% | about 0.6% |

  Running LAN mode only during class (say 6 hours a day) cuts these to about a quarter. Every New PIN also
  restarts the server-wide budget, which gives a guesser another 50 tries against the new PIN (another 0.005%).
- **New PIN** calls `gate.rotate(revoke_tokens=True)`: a new PIN, every token revoked, and every phone's live
  Socket.IO connection closed. The server-wide budget restarts; a device's own lockout keeps running, so a script
  that locked itself out stays locked out. **Disconnect all phones** calls `gate.revoke_all()` and closes the
  same connections without changing the PIN.
- **The cookie** (`sar_lan`) carries a 256-bit random token. The server keeps only its SHA-256 hash, in memory,
  for 12 hours of wall-clock time, so a laptop that sleeps doesn't stretch it. The cookie is `HttpOnly`,
  `SameSite=Lax` and `Path=/`, and not `Secure` because the LAN is plain HTTP. Restarting the assistant signs
  every device out.
- **The laptop itself never needs the PIN.** Requests from a loopback address (`127.0.0.1`, `::1`) pass. Open the
  dashboard on the laptop as `http://127.0.0.1:5050`. Through its own LAN address the laptop counts as just
  another device.
- **DNS rebinding is blocked.** The `Host` header must be an IP address, `localhost` or a name listed in
  `SAR_LAN_HOSTNAMES` (for example `SAR_LAN_HOSTNAMES=my-laptop.local`). Without this check, a web page that
  points its own domain at `127.0.0.1` could borrow the laptop's loopback trust. List only names on your own
  network: whoever controls a listed name could use it for exactly that.
- The existing same-origin guards stay: `block_cross_origin` for POSTs, and Socket.IO's
  `cors_allowed_origins=None`.
- **Known limits.**
  - **Open Wi-Fi.** On a network without a password anyone can sniff the plain HTTP traffic, PIN and cookie
    included. WPA2/WPA3 networks encrypt each device's traffic, so use one of those, never public Wi-Fi.
  - **A device with the cookie** has the dashboard's full powers: files, modes and quizzes.
  - **Blocking new joins.** A script spread over two or more addresses (or one that keeps changing its address)
    can use up the server-wide budget and keep the join form locked for everyone for up to an hour at a time. Joined devices are not affected. New PIN
    restarts the budget but also signs everyone out, so the real fix is a network that person isn't on (a phone
    hotspot).
  - **The QR link is a GET.** `GET /join?pin=` counts a guess only when it looks like a page load (a QR scan or a
    typed address), so an `<img>` tag on some other web page can't burn guesses. On plain HTTP browsers don't
    send `Sec-Fetch-*` headers, so this rests on the `Accept` header, which a script can fake. The per-device
    limit still applies: such a page can lock out only the device it runs on, though its misses also count
    toward the server-wide budget.
  - **Never put the assistant behind a tunnel, reverse proxy or port forwarder** (ngrok, cloudflared, VS Code or
    SSH port forwarding, a local nginx). They connect from `127.0.0.1`, which the gate trusts. One that rewrites
    `Host` to `localhost` or `127.0.0.1` (`ngrok --host-header=rewrite`, for example) would open the whole
    dashboard to anyone without a PIN.

## Wiring spec (for the integrator)

Everything below goes into files other agents own: `config.py`, `app.py`, `templates/*` and
`static/js/dashboard/*`. Each step names the `lan.py` function to use. Ship tests with each step (see the end).

1. **Settings (`config.py`).**
   - `sar_lan = os.environ.get("SAR_LAN") == "1"`.
   - `host` defaults to `"0.0.0.0" if sar_lan else "127.0.0.1"`. An explicit `SAR_HOST` still wins.
   - **`lan = is_lan_mode(sar_lan=sar_lan, bind_host=host)`.** This turns LAN mode on for `SAR_LAN=1` and for
     **any non-loopback bind**: `SAR_HOST=0.0.0.0` (or `::`, `""`, a LAN address or a host name) without
     `SAR_LAN` must never listen ungated. When `lan` is on only because of `SAR_HOST`, log that the PIN gate is
     on. When `SAR_LAN=1` but `SAR_HOST` is loopback, log a warning, because phones can't reach it.
   - `lan_hostnames = parse_hostnames(os.environ.get("SAR_LAN_HOSTNAMES"))`: extra `Host` names to accept, such
     as `my-laptop.local`. Pass them as `extra_names` everywhere below.

2. **One gate per server (`create_app`).** When `settings.lan` is on, create `gate = JoinPin()` and
   `phones = PhoneSessions()`, and store them in `app.config["SAR_JOIN_PIN"]` and `app.config["SAR_PHONES"]`.
   Also call `mimetypes.add_type("application/manifest+json", ".webmanifest")`, because Windows' MIME table may
   not have it.

3. **`before_request` gate**, registered after `block_cross_origin`, and an `after_request` hook for the headers:

   ```python
   from biobench_sar.lan import COOKIE_NAME, LAN_MODE_HEADER, is_allowed_host, is_request_allowed


   @app.before_request
   def lan_gate() -> Any:
       if not settings.lan:
           return None
       names = settings.lan_hostnames
       if not is_allowed_host(request.host, names):
           return jsonify({"error": "unknown host"}), 400  # refused before /join, so this can't loop
       if is_request_allowed(
           gate,
           remote_addr=request.remote_addr,
           host=request.host,
           path=request.path,
           token=request.cookies.get(COOKIE_NAME),
           extra_names=names,
       ):
           return None
       if request.path.startswith("/api/") or request.method != "GET":
           return jsonify({"error": "join with the PIN first"}), 401
       return redirect("/join")


   @app.after_request
   def lan_headers(response: Response) -> Response:
       if settings.lan:
           response.headers[LAN_MODE_HEADER] = "1"  # the service worker retires itself without it
           if request.path.startswith("/api/") or request.path == "/join":
               response.headers["Cache-Control"] = "no-store"
       return response
   ```

   Pass `request.remote_addr`, the TCP peer. Never trust `X-Forwarded-For`: there is no proxy, and there must
   never be one (see Known limits).

4. **`GET /join`** shows the PIN form and sets the cookie on success. `POST /join` (form field `pin`) handles a
   typed PIN the same way.
   - A local request (`is_local_request`) redirects to `/dashboard` at once, since the laptop never needs a PIN.
   - A `?pin=` (from the QR code) counts only when
     `looks_like_navigation(accept=request.headers.get("Accept"), sec_fetch_mode=request.headers.get("Sec-Fetch-Mode"), sec_fetch_site=request.headers.get("Sec-Fetch-Site"))`
     is true. Otherwise show the form without trying the PIN.
   - Call `token = gate.redeem(pin, request.remote_addr)`. On success, call
     `response.set_cookie(COOKIE_NAME, token, **cookie_kwargs())` and send a **303 to `/dashboard`**, which
     takes the PIN out of the address bar and the history.
   - On failure, show the form again with "That PIN didn't work". While
     `gate.locked_for(request.remote_addr) > 0`, show "Too many wrong PINs. Try again in N minutes." (or
     seconds, under a minute).
   - Send `Cache-Control: no-store` and `Referrer-Policy: no-referrer` on every `/join` response.
   - Add a new `templates/join.html`: a light page in the dashboard's style. Give the input
     `inputmode="numeric" autocomplete="one-time-code" maxlength="9"`; `verify()` ignores spaces and dashes.

5. **Socket.IO is gated at the WSGI layer.** Flask's `before_request` never runs for `/socket.io/`, because
   python-socketio answers before Flask. A check inside `on_connect` alone is not enough: with
   `always_connect=True` python-socketio adds the sid to the namespace before `on_connect` runs, and Engine.IO
   handles each packet on its own thread, so broadcasts (`transcript`, `speak`, `interaction`,
   `documents_changed`) can reach a device while its refused connect is still running. So, right after
   `socketio = SocketIO(app, ...)`:

   ```python
   if settings.lan:
       app.wsgi_app = SocketIOGate(app.wsgi_app, gate, extra_names=settings.lan_hostnames)
   ```

   `SocketIO(app)` put its own middleware in `app.wsgi_app`, so `SocketIOGate` runs first and answers 401 to the
   handshake and to every later polling request without a live token. As defence in depth, check again at the top
   of **every** `@socketio.on` handler, the ones added later included, and track phone connections so they can
   be closed:

   ```python
   def phone_allowed() -> bool:
       return not settings.lan or is_request_allowed(
           gate,
           remote_addr=request.remote_addr,
           host=request.host,
           path="/socket.io/",
           token=request.cookies.get(COOKIE_NAME),
           extra_names=settings.lan_hostnames,
       )


   @socketio.on("connect")
   def on_connect(_auth: Any = None) -> None:
       if not phone_allowed():
           raise ConnectionRefusedError("join with the PIN first")
       if settings.lan and not is_local_request(request.remote_addr):
           phones.add(request.sid)
       ...


   @socketio.on("disconnect")
   def on_disconnect(*_args: Any) -> None:
       phones.discard(request.sid)


   @socketio.on("text_query")
   def on_text_query(data: Any) -> None:
       if not phone_allowed():
           disconnect()
           return
       ...
   ```

   `flask.request` in a handler is the connection's handshake request, cookies included, so a revoked token
   fails here even on a WebSocket that is still open.

6. **Dashboard card "Use a phone or tablet"**, shown only when LAN mode is on and the viewer is the laptop.
   - **Local-only endpoints.** All three answer 403 unless `is_local_request(request.remote_addr)`, so a joined
     phone never sees the PIN. `/api/*` is `no-store` (step 3), so the PIN never lands in a cache.
     - `GET /api/lan` returns `{"enabled": true, "pin": gate.pin, "locked_for": gate.server_locked_for(),
       "phones": len(phones), "urls": [join_url(ip, settings.port, gate.pin) for ip in local_ip_addresses()]}`.
       With LAN mode off it returns `{"enabled": false}`.
     - `POST /api/lan/pin` (**New PIN**) calls `gate.rotate(revoke_tokens=True)`, then closes every phone's
       connection with `for sid in phones.take_all(): socketio.server.disconnect(sid, namespace="/")`, and
       returns the same body as `GET /api/lan`.
     - `POST /api/lan/revoke` (**Disconnect all phones**) calls `gate.revoke_all()` and closes the connections the
       same way. The PIN stays.
   - **The card.** Show the QR of `urls[0]`, the PIN in large type grouped as `123 456`, the address
     (`urls[0]` without the `?pin=`) for typing, any other addresses in small type, how many phones are
     connected, and two buttons: "New PIN" (with the hint "signs every phone out") and "Disconnect all phones".
     If `urls` is empty, say "This laptop isn't on a network. Join the classroom Wi-Fi." If `locked_for` is
     above 0, say "Too many wrong PINs: nobody can join for N minutes."
   - **QR code.** Load `{{ url_for('static', filename='vendor/qrcode.min.js') }}` as a classic script (it
     defines a global `qrcode`), then render:

     ```js
     const qr = qrcode(0, "M");  // 0 = smallest version that fits
     qr.addData(url);
     qr.make();
     box.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 4, scalable: true });
     ```

7. **Startup log (`main`).** When LAN mode is on, print the laptop's address as `http://127.0.0.1:<port>`
   (`0.0.0.0` can't be clicked). Then print `Phones  <join_url without ?pin>  PIN 123 456` for each address, a
   line about the firewall prompt and one saying "Disconnect any VPN if phones can't connect".

8. **Templates.** Add these to `precharge.html` and `join.html`. The projector page doesn't need them.

   ```html
   <link rel="manifest" href="{{ url_for('static', filename='manifest.webmanifest') }}">
   <meta name="theme-color" content="#23497d">
   <link rel="icon" href="{{ url_for('static', filename='icons/icon.svg') }}" type="image/svg+xml">
   <link rel="apple-touch-icon" href="{{ url_for('static', filename='icons/apple-touch-icon.png') }}">
   <meta name="apple-mobile-web-app-title" content="Whiteboard">
   ```

9. **Service worker, in LAN mode only.** It runs only on HTTPS or `localhost`, so today it helps the laptop alone;
   skipping it is fine. If you wire it:
   - **Serve it from the root, and only in LAN mode.** With LAN mode off, `/sw.js` answers 404, so nothing new
     can register. In LAN mode:

     ```python
     @app.get("/sw.js")
     def service_worker() -> Any:
         if not settings.lan:
             abort(404)
         response = send_from_directory(app.static_folder, "sw.js", mimetype="text/javascript", max_age=0)
         response.headers["Cache-Control"] = "no-cache"
         response.headers["Service-Worker-Allowed"] = "/"
         return response
     ```

   - **Register it only in LAN mode**, and clean up outside it. The template passes `lan` to the page:

     ```js
     if ("serviceWorker" in navigator) {
       if (LAN) {
         navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {});
       } else {
         // Kill switch: a worker left over from an earlier LAN-mode run on this port.
         navigator.serviceWorker.getRegistrations().then((rs) => rs.forEach((r) => r.unregister())).catch(() => {});
         if (window.caches) caches.keys().then((ns) => ns.filter((n) => n.startsWith("sar-")).forEach((n) => caches.delete(n))).catch(() => {});
       }
     }
     ```

     `serviceWorker` is missing on plain-HTTP phones, so they skip all of this.
   - **The worker's own dev-safety rule** covers pages where that script never runs. It stores only responses
     carrying `X-SAR-LAN: 1` (step 3). The first unmarked response from the laptop means LAN mode is off, so the
     worker deletes every `sar-*` cache, unregisters itself and stops intercepting. Edited JS and CSS then show up
     on the next reload.
   - **By hand:** DevTools → Application → Service workers → Unregister, or "Clear site data" for
     `http://127.0.0.1:5050`.
   - When the cached shell list changes, bump `CACHE_VERSION` in `sw.js`. Activation deletes every other `sar-*`
     cache.
   - What it does:
     - Page loads are network first. The cached page appears only when the laptop can't be reached.
     - `/static/*` is network first until the worker has seen a LAN-mode response since it started, and
       stale-while-revalidate after that.
     - **`/api/*` is network-only and never stored**: live data must never be stale, and `GET /api/lan` carries
       the PIN. The same goes for Socket.IO, `/join`, `/tts/*`, `/calibration.png`, non-GET and range requests,
       other origins, redirects (the gate sending a device to `/join`) and `no-store` responses.

10. **Tests to ship with the wiring.**
    - **LAN mode off:** nothing changes, and `/sw.js` returns 404.
    - **LAN mode on:**
      - Use `client.get(..., environ_base={"REMOTE_ADDR": "192.168.1.30"})` for a phone.
      - `/dashboard` redirects to `/join`, and `/api/status` returns 401.
      - **`SAR_HOST=0.0.0.0` without `SAR_LAN`**: a phone gets 401 on `/api/status`.
      - `/join?pin=<right>` with `Accept: text/html` sets `sar_lan` and returns 303 to `/dashboard`; with the
        cookie, `/dashboard` returns 200. The same URL with `Accept: image/*` doesn't count a guess.
      - 5 wrong PINs from one phone lock that phone; a second phone can still join.
      - `Host: evil.example` returns 400; with `SAR_LAN_HOSTNAMES=my-laptop.local`, `Host: my-laptop.local`
        passes the host check.
      - Loopback passes without a cookie.
      - `/api/lan` returns 403 from a phone; every `/api/*` response has `Cache-Control: no-store` and
        `X-SAR-LAN: 1`.
      - After `POST /api/lan/pin` or `POST /api/lan/revoke`, a phone's cookie gets 401.
      - `/sw.js` has `Service-Worker-Allowed: /`.
      - **Socket.IO:** `socketio.test_client` calls python-socketio directly and skips the WSGI layer, so test
        `SocketIOGate` with `app.test_client().get("/socket.io/?EIO=4&transport=polling", ...)`: 401 for a phone
        without the cookie, and an Engine.IO `0{...` open packet with it. For the handler checks, use
        `socketio.test_client(app, flask_test_client=client)` so the cookie carries over; a refused connect
        leaves `is_connected()` false.

`tests/test_lan.py` has a reference wiring of steps 1 to 6 and 9 that exercises these rules end to end.
