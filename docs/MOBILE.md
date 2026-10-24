# Phones and tablets: the remote dashboard

The assistant needs the laptop, because the camera, the models, the microphone and the projector all live
there. A phone or tablet can't run it, but on the same Wi-Fi it can open the laptop's dashboard and drive it
remotely. The laptop still does all the work.

**Phone mode (LAN mode)** is off by default. It is on for `SAR_LAN=1` and for **any** bind other devices could
reach (a `SAR_HOST` other than `127.0.0.1`, `localhost` or `::1`), and then every device but the laptop needs a
six-digit PIN. The pieces:

| File | What it is |
| --- | --- |
| `biobench_sar/lan.py` | This machine's LAN addresses, `join_url`, the `JoinPin` PIN and cookie gate, the host and loopback checks, `SocketIOGate` and `PhoneSessions` |
| `biobench_sar/config.py` | `Settings.lan` and `Settings.lan_hostnames` |
| `biobench_sar/app.py` | The gate (`before_request` + `SocketIOGate`), `/join`, `/api/lan*`, `/sw.js`, the response headers and the startup banner |
| `biobench_sar/templates/join.html` | The PIN form a phone sees first (English or Spanish) |
| `biobench_sar/static/js/dashboard/phone.js` | The laptop's **Use a phone or tablet** card under **More tools**: QR code, PIN, phones connected, New PIN, Disconnect all phones |
| `biobench_sar/static/manifest.webmanifest`, `static/icons/` | Add to Home Screen: name, colours, icons (regenerate them with `scripts/make_icons.py`) |
| `biobench_sar/static/sw.js` | A small service worker for LAN mode that makes reloads faster on the laptop (see its limits below) |
| `biobench_sar/static/vendor/qrcode.min.js` | qrcode-generator 1.4.4 (MIT), unmodified from cdnjs. Its licence is in `qrcode.LICENSE.txt` |

## What works on a phone

- **Asking questions by typing.** To ask by voice, use the keyboard's **dictation microphone** (the mic key on
  the iPhone keyboard, or voice typing in Gboard). The phone turns the speech into text, so the browser never
  needs microphone permission.
- **Seeing the answers** and the assistant's status as they happen, through the same Socket.IO updates the
  laptop dashboard gets.
- **Changing the mode** (step-by-step help, similar problems or quiz) and **starting or stopping a quiz**.
- **Choosing the study files**: tick or untick worksheets and notes. Adding a file from the phone uses the same
  picker, and on most phones it can take a photo of a worksheet.
- **English or Spanish**: the join page follows the phone's language, and the dashboard has the same
  English · Español · Auto switch.
- **Add to Home Screen** gives the phone an icon that opens the dashboard full screen.

## What doesn't work, and the workarounds

- **The phone's microphone inside the page.** Browsers only allow `getUserMedia` on HTTPS or `localhost`, and
  the laptop serves plain HTTP on the Wi-Fi. Use keyboard dictation instead. The dashboard's **Talk** button on
  a phone starts the **laptop's** microphone, which is useful when the student is at the board.
- **Spoken answers** play in the room through the laptop and projector, not on the phone.
- **Using the phone as the camera.** This isn't possible through the browser. As an optional workaround, make
  the phone a webcam for the laptop, then point `SAR_CAMERA_INDEX` at it and check it with
  `scripts/hardware_check.py` (it needs at least 28 FPS):
  - **Continuity Camera** (iPhone with iOS 16+ and a Mac with macOS 13+): the iPhone appears as an ordinary
    camera.
  - **DroidCam** (Android or iPhone with Windows, macOS or Linux): an app on the phone plus a virtual-webcam
    client on the laptop.
- **Offline use and a true installed app on the phone.** Service workers and installable web apps need HTTPS
  too, so over plain-HTTP Wi-Fi the phone gets a home-screen shortcut without offline caching. The service worker
  only runs on the laptop (`http://127.0.0.1` counts as secure).
- **iPhone home-screen icon.** It keeps its own cookies, separate from Safari's, so the first launch from the
  icon shows the PIN form once. Type the PIN.

## How a teacher turns it on

1. **Same network.** The laptop and the phones join the same Wi-Fi. Guest and campus networks often use
   *client isolation*, which stops devices from reaching each other. If the phone can't load the page, use a
   phone hotspot or a router of your own.
2. **Start the assistant with `SAR_LAN=1`** (or add `SAR_LAN=1` to `sar.env`; `true`, `yes` and `on` work too).
   It then listens on every interface (`0.0.0.0`) unless `SAR_HOST` says otherwise:

   ```bash
   SAR_LAN=1 .venv/bin/python -m biobench_sar
   ```

   ```powershell
   $env:SAR_LAN='1'; .\.venv\Scripts\python.exe -m biobench_sar
   ```

   The first time, the system asks whether Python (or the app) may accept incoming connections. On Windows,
   allow **Private networks**, and on macOS click **Allow**. Setting `SAR_HOST` to anything other than
   `127.0.0.1`, `localhost` or `::1` (for example `0.0.0.0`) also turns LAN mode on, PIN included.
3. **The startup banner** prints, for each address phones can reach,
   `Phones  http://<laptop-ip>:5050/join   PIN 123 456`, then a line about the firewall prompt and one saying
   "Disconnect any VPN if phones can't connect".
4. **On the laptop**, open `http://127.0.0.1:5050/dashboard` as usual. Under **More tools**, the **Use a phone
   or tablet** card shows a QR code, the PIN, the address to type, any other addresses, and how many phones are
   connected.
5. **On the phone**, scan the QR code with the camera app. It opens the dashboard and the phone is in. Without
   the QR code, type the address (`http://<laptop-ip>:5050/join`) and then the PIN.
6. A device that has joined stays in for 12 hours or until the assistant restarts. **Every restart makes a new
   PIN.** On the card:
   - **New PIN** makes a new PIN and signs every phone out. They scan the new QR code to come back. Use it if
     the PIN has leaked.
   - **Disconnect all phones** signs every phone out and closes their live connections, but keeps the PIN. Use
     it at the end of a class.
7. **When class is over**, restart the assistant without `SAR_LAN=1`. A server left in LAN mode overnight gives
   a guesser more time (see the odds below).

### If a phone can't connect

- **It can't load the page at all.** Check that both are on the same Wi-Fi and that the firewall prompt was
  allowed. Guest and campus networks often isolate devices, so use a phone hotspot or your own router. On a
  standard (non-admin) Windows school account, allowing the firewall prompt may need IT.
- **Disconnect any VPN on the laptop.** A full-tunnel VPN makes the laptop report its VPN address, which the QR
  code then points to and phones can't reach. Corporate and campus VPNs often do this.
- **Try the other addresses** listed in small type under the QR code, if there are any.
- **"Too many wrong PINs"** means that device has to wait. Joined devices keep working.
- **The card says the laptop isn't on a network**: join the classroom Wi-Fi on the laptop.

## Security model

- **Off by default.** With `SAR_LAN` unset and `SAR_HOST` unset or loopback (`127.0.0.0/8`, `::1`,
  `localhost`), the server listens on this machine only, so nothing on the network can reach it.
- **Every network bind is gated.** LAN mode is on whenever `SAR_LAN=1` **or** `SAR_HOST` is anything else
  (`0.0.0.0`, `::`, an empty string, a LAN address or a host name). The server can't listen on the network
  without the PIN gate. Anyone on the same network can then reach the server, but without a valid cookie they
  get only the join form and the static files (CSS, JS, icons). Everything else needs the cookie: pages,
  `/api/*`, `/tts/*`, camera frames and Socket.IO.
- **The PIN** has six digits drawn with `secrets` and is compared in constant time (`hmac.compare_digest`).
  It lives in memory only. Every `/api/*` response is `Cache-Control: no-store`, in and out of LAN mode, and the
  service worker never stores them, so the PIN that `GET /api/lan` returns is never written to a cache.
- **Guessing limits.** Wrong PINs count **per device**, keyed on the TCP peer address (for IPv6 the `/64`,
  because one phone can use many addresses):
  - 5 wrong PINs within a minute lock that device for 1 minute. Each further lockout doubles (2, 4, 8, ...
    minutes) up to **1 hour**. A right PIN, or a day without wrong PINs, starts that device over.
  - On top of that, **all devices together** get at most **50 wrong PINs an hour**. Past that, nobody can join
    until the oldest miss is an hour old. Devices that have already joined keep working. One device on its own
    makes at most 30 wrong guesses in an hour, so it can never reach this cap alone.
  - Guesses made while locked out are refused and don't count.
- **The odds, honestly.** A guess is right one time in a million. The table assumes a script guessing as fast
  as the limits allow, the assistant running around the clock in LAN mode, and nobody pressing New PIN:

  | Who is guessing | Guesses a day | Chance per day | Chance over a 5-day school week |
  | --- | --- | --- | --- |
  | One device, one address | about 145 on day 1, then 120 | about 0.015% | about 0.06% |
  | A script that keeps changing its address (the server-wide cap) | 1,200 | about 0.12% | about 0.6% |

  Running LAN mode only during class (say 6 hours a day) cuts these to about a quarter. Every New PIN also
  restarts the server-wide budget, which gives a guesser another 50 tries against the new PIN (another
  0.005%).
- **New PIN** (`POST /api/lan/rotate`) calls `gate.rotate(revoke_tokens=True)`: a new PIN, every token
  revoked, and every phone's live Socket.IO connection closed. The server-wide budget restarts, but a device's
  own lockout keeps running, so a script that locked itself out stays locked out. **Disconnect all phones**
  (`POST /api/lan/disconnect`) calls `gate.revoke_all()` and closes the same connections without changing the
  PIN.
- **The cookie** (`sar_lan`) carries a 256-bit random token. The server keeps only its SHA-256 hash, in memory,
  for 12 hours of wall-clock time, so a laptop that sleeps doesn't stretch it. The cookie is `HttpOnly`,
  `SameSite=Lax`, `Path=/` and `Max-Age` 12 hours, and not `Secure` because the LAN is plain HTTP. Restarting
  the assistant signs every device out.
- **The laptop itself never needs the PIN.** A request is local when its TCP peer is a loopback address
  (`127.0.0.1`, `::1`) **and** its `Host` is a loopback or allowed name. Open the dashboard on the laptop as
  `http://127.0.0.1:5050`. Through its own LAN address, the laptop counts as just another device.
- **DNS rebinding is blocked.** The `Host` header must be an IP address, `localhost` or a name listed in
  `SAR_LAN_HOSTNAMES` (for example `SAR_LAN_HOSTNAMES=my-laptop.local`). Any other `Host` gets a 400. Without
  this check, a web page that points its own domain at `127.0.0.1` could borrow the laptop's loopback trust.
  List only names on your own network, because whoever controls a listed name could use it for exactly that.
- **`/join` sends `Referrer-Policy: same-origin`**, so the `?pin=` in a QR link never reaches another site in a
  `Referer`. It is not `no-referrer`: with that, browsers send `Origin: null` on the PIN form's POST, and the
  same-origin guard (`block_cross_origin`, which stays on for every POST) would refuse it. The guard accepts the
  same-origin `POST /join`. Socket.IO keeps `cors_allowed_origins=None`.
- **Known limits.**
  - **Open Wi-Fi.** On a network without a password, anyone can sniff the plain HTTP traffic, PIN and cookie
    included. WPA2/WPA3 networks encrypt each device's traffic, so use one of those, never public Wi-Fi.
  - **A device with the cookie** has the dashboard's full powers: files, modes and quizzes. It never sees the
    PIN, because `/api/lan` answers only the laptop.
  - **Blocking new joins.** A script spread over two or more addresses (or one that keeps changing its
    address) can use up the server-wide budget and keep the join form locked for everyone for up to an hour at
    a time. Joined devices are not affected. New PIN restarts the budget but also signs everyone out, so the
    real fix is a network that person isn't on (a phone hotspot).
  - **The QR link is a GET.** `GET /join?pin=` counts a guess only when it looks like a page load (a QR scan or
    a typed address), so an `<img>` tag on some other web page can't burn guesses. On plain HTTP, browsers don't
    send `Sec-Fetch-*` headers, so this rests on the `Accept` header, which a script can fake. The per-device
    limit still applies: such a page can lock out only the device it runs on, though its misses also count
    toward the server-wide budget.
  - **Never put the assistant behind a tunnel, reverse proxy or port forwarder** (ngrok, cloudflared, VS Code
    or SSH port forwarding, a local nginx). They connect from `127.0.0.1`, which the gate trusts. One that
    rewrites `Host` to `localhost` or `127.0.0.1` (`ngrok --host-header=rewrite`, for example) would open the
    whole dashboard to anyone without a PIN.

## How the server does it (for developers)

Each rule names the `lan.py` function it uses. Keep all of them when you change the server, and add a test for
any new route or socket handler.

1. **Settings (`config.py`).**
   - `SAR_LAN` is read as on for `1`, `true`, `yes` or `on`. With it, `host` defaults to `0.0.0.0`. An explicit
     `SAR_HOST` still wins.
   - **`lan = is_lan_mode(sar_lan=..., bind_host=host)`**: on for `SAR_LAN` and for **any non-loopback bind**.
     When `lan` is on only because of `SAR_HOST`, the log says the PIN gate is on. `SAR_LAN=1` with a loopback
     `SAR_HOST` logs a warning, because phones can't reach it.
   - `lan_hostnames = parse_hostnames(SAR_LAN_HOSTNAMES)`: extra `Host` names to accept, passed as
     `extra_names` everywhere.
2. **One gate per server (`create_app`).** `JoinPin()` and `PhoneSessions()`, kept in
   `app.config["SAR_JOIN_PIN"]` / `app.config["SAR_PHONES"]`. `.webmanifest` is registered as
   `application/manifest+json`, because Windows' MIME table may not have it.
3. **`before_request` gate**, after `block_cross_origin`, in LAN mode only:
   - `is_allowed_host(request.host, names)` fails → 400 `unknown host` (before `/join`, so this can't loop).
   - `is_request_allowed(gate, remote_addr=request.remote_addr, host=request.host, path=request.path,
     token=request.cookies.get(COOKIE_NAME), extra_names=names)` passes → the request goes on.
   - Otherwise `/api/*` and non-GET requests get 401 `join with the PIN first`, and pages get a redirect to
     `/join`.
   - Always pass `request.remote_addr`, the TCP peer. Never trust `X-Forwarded-For`: there is no proxy, and
     there must never be one.
4. **Headers (`after_request`).** `/api/*` and `/join` get `Cache-Control: no-store` in every mode. `/join`
   gets `Referrer-Policy: same-origin`. In LAN mode every response carries `X-SAR-LAN: 1` (`LAN_MODE_HEADER`), the
   mark the service worker looks for.
5. **`/join`.**
   - A local request, or any request with LAN mode off, gets a 303 to `/dashboard`.
   - `GET /join` renders `templates/join.html` with `error` (bool), `language` (the phone's `Accept-Language`,
     English or Spanish) and the lockout message while `gate.locked_for(request.remote_addr) > 0`.
   - `GET /join?pin=` (the QR link) tries the PIN only when `looks_like_navigation(accept=...,
     sec_fetch_mode=..., sec_fetch_site=...)` is true. A device that already has a valid cookie goes straight to
     `/dashboard`.
   - `POST /join` (form field `pin`) and a counted `?pin=` call `gate.redeem(pin, request.remote_addr)`. On
     success the server calls `response.set_cookie(COOKIE_NAME, token, **cookie_kwargs())` and sends a **303 to
     `/dashboard`**, which takes the PIN out of the address bar and the history. On failure, the form comes back
     with the error ("That PIN didn't work…", or "Too many wrong PINs. Try again in N minutes.").
   - The input is `inputmode="numeric" autocomplete="one-time-code" maxlength="9"`, and `verify()` ignores
     spaces and dashes.
6. **Socket.IO is gated at the WSGI layer.** Flask's `before_request` never runs for `/socket.io/`, because
   python-socketio answers before Flask. A check inside `on_connect` alone is not enough: with
   `always_connect=True` python-socketio adds the sid to the namespace before `on_connect` runs, so broadcasts
   could reach a device while its refused connect is still running. So, right after `SocketIO(app, ...)`, in
   LAN mode, `app.wsgi_app = SocketIOGate(app.wsgi_app, gate, extra_names=settings.lan_hostnames)` answers 401 to
   the handshake and to every later polling request without a live token. As defence in depth, `on_connect`
   refuses the connection (`ConnectionRefusedError`) and every `@socketio.on` handler checks again, and phone
   sids are tracked in `PhoneSessions` so New PIN and Disconnect can close them
   (`socketio.server.disconnect(sid, namespace="/")`). `flask.request` in a handler is the connection's
   handshake request, cookies included, so a revoked token fails there even on a WebSocket that is still open.
7. **Local-only endpoints** (403 unless `is_local_request(request.remote_addr)`, so a joined phone never sees
   the PIN):

   | Method | Path | Result |
   | --- | --- | --- |
   | GET | `/api/lan` | `{"enabled": bool, "urls": [join_url(ip, port, pin) for ip in local_ip_addresses()], "pin": "123456" \| null, "phones": int}` (plus `locked_for`: `gate.server_locked_for()`, the seconds until anyone may join again). With LAN mode off: `enabled` false, `pin` null |
   | POST | `/api/lan/rotate` | New PIN: `gate.rotate(revoke_tokens=True)`, then every phone socket is disconnected. Returns the `GET /api/lan` body |
   | POST | `/api/lan/disconnect` | Disconnect all phones: `gate.revoke_all()`, then every phone socket is disconnected. The PIN stays |

   The groundwork's names for the two buttons, `POST /api/lan/pin` and `POST /api/lan/revoke`, were never
   built: use `/api/lan/rotate` and `/api/lan/disconnect`. With LAN mode off, both answer 409.

8. **The card** (`static/js/dashboard/phone.js`) polls `GET /api/lan` on the laptop and draws the QR of
   `urls[0]` with `static/vendor/qrcode.min.js` (black on white whatever the theme, or phones may not read it),
   the PIN grouped as `123 456`, `urls[0]` without the `?pin=` for typing, the other addresses in small type,
   the phone count, and the two buttons, each with a confirmation.
9. **Templates** receive `lan_enabled` (bool). In LAN mode the dashboard and the join page link the manifest,
   the icons and the `theme-color`. The projector page doesn't need them.
10. **Startup banner (`main`).** The laptop's address is printed as `http://127.0.0.1:<port>` (`0.0.0.0` can't
    be clicked). In LAN mode, a `Phones  <join url without ?pin>  PIN 123 456` line follows for each address,
    then the firewall and VPN lines.
11. **Service worker, in LAN mode only.** It runs only on HTTPS or `localhost`, so today it helps the laptop
    alone.
    - **`/sw.js` is served only in LAN mode** (`Cache-Control: no-cache`, `Service-Worker-Allowed: /`). With
      LAN mode off it answers 404, so nothing new can register.
    - **The page registers it only in LAN mode** (`navigator.serviceWorker.register("/sw.js", { scope: "/" })`).
      Otherwise it unregisters any leftover worker (`navigator.serviceWorker.getRegistrations()`, then
      `unregister()` on each) and deletes the `sar-*` caches: a kill switch for a worker from an earlier LAN-mode
      run on this port. `serviceWorker` is
      missing on plain-HTTP phones, so they skip all of this.
    - **The worker's own dev-safety rule** covers pages where that script never runs. It stores only responses
      carrying `X-SAR-LAN: 1`. The first unmarked response from the laptop means LAN mode is off, so the
      worker deletes every `sar-*` cache, unregisters itself and stops intercepting. Edited JS and CSS then
      show up on the next reload.
    - **By hand:** DevTools → Application → Service workers → Unregister, or "Clear site data" for
      `http://127.0.0.1:5050`.
    - When the cached shell list changes, bump `CACHE_VERSION` in `sw.js`. Activation deletes every other
      `sar-*` cache.
    - What it does:
      - Page loads are network first. The cached page appears only when the laptop can't be reached.
      - `/static/*` is network first until the worker has seen a LAN-mode response since it started, and
        stale-while-revalidate after that.
      - **`/api/*` is network-only and never stored**: live data must never be stale, and `GET /api/lan`
        carries the PIN. The same goes for Socket.IO, `/join`, `/tts/*`, `/calibration.png`, non-GET and range
        requests, other origins, redirects (the gate sending a device to `/join`) and `no-store` responses.

### Rules the tests hold the server to

- **LAN mode off:** nothing changes, `/sw.js` returns 404, and `/api/*` is still `no-store`.
- **LAN mode on** (a phone is `client.get(..., environ_base={"REMOTE_ADDR": "192.168.1.30"})`):
  - `/dashboard` redirects to `/join`, and `/api/status` returns 401. The same holds for
    **`SAR_HOST=0.0.0.0` without `SAR_LAN`**.
  - `POST /join` with the right PIN sets `sar_lan` and returns 303 to `/dashboard`, and with the cookie
    `/dashboard` returns 200. A same-origin form POST (with an `Origin` header) passes the cross-origin guard.
  - `/join?pin=<right>` with `Accept: text/html` joins. The same URL with `Accept: image/*` doesn't count a
    guess.
  - `/join` responses carry `Referrer-Policy: same-origin`.
  - 5 wrong PINs from one phone lock that phone, and a second phone can still join.
  - `Host: evil.example` returns 400. With `SAR_LAN_HOSTNAMES=my-laptop.local`, `Host: my-laptop.local` passes
    the host check.
  - Loopback passes without a cookie.
  - `/api/lan` returns 403 from a phone. Every `/api/*` response has `Cache-Control: no-store` and
    `X-SAR-LAN: 1`.
  - After `POST /api/lan/rotate` or `POST /api/lan/disconnect`, a phone's cookie gets 401.
  - `/sw.js` has `Service-Worker-Allowed: /`.
  - **Socket.IO:** `socketio.test_client` calls python-socketio directly and skips the WSGI layer, so
    `SocketIOGate` is tested with `app.test_client().get("/socket.io/?EIO=4&transport=polling", ...)`: 401 for a
    phone without the cookie, and an Engine.IO `0{...` open packet with it. The handler checks use
    `socketio.test_client(app, flask_test_client=client)` so the cookie carries over, and a refused connect
    leaves `is_connected()` false.
