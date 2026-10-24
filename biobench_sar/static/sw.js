// Service worker for the dashboard in LAN mode. Browsers run service workers only on https:// or localhost, so
// phones on plain-HTTP Wi-Fi skip it and it helps the laptop alone; every page must work without it.
//   page loads (navigations)  network first; the cached page only when the laptop can't be reached
//   /static/*                 network first until this worker has seen LAN mode, then stale-while-revalidate
//   everything else           never answered from or stored in CacheStorage: /api/* (live data and the PIN),
//                             Socket.IO, /join, /tts/*, non-GET and range requests, other sites
// Dev safety: in LAN mode the server marks every response with "X-SAR-LAN: 1". An unmarked response from this
// origin means LAN mode is off (an ordinary run on the same port), so the worker deletes its caches, unregisters
// itself and stops intercepting: edited JS and CSS show up on the next reload. Nothing unmarked is ever stored.
// Serve it from /sw.js (scope "/") so it covers /dashboard. Bump CACHE_VERSION when SHELL changes; activate
// deletes every other sar-* cache.
"use strict";

const CACHE_VERSION = "v2";
const CACHE = `sar-shell-${CACHE_VERSION}`;
const LAN_HEADER = "X-SAR-LAN";

const SHELL = [
  "/dashboard",
  "/static/css/dashboard.css",
  "/static/js/vendor/socket.io.min.js",
  "/static/js/dashboard/main.js",
  "/static/js/dashboard/api.js",
  "/static/js/dashboard/ask.js",
  "/static/js/dashboard/camera.js",
  "/static/js/dashboard/files.js",
  "/static/js/dashboard/log.js",
  "/static/js/dashboard/profile.js",
  "/static/js/dashboard/status.js",
  "/static/js/dashboard/tabs.js",
  "/static/js/dashboard/tutor.js",
  "/static/vendor/qrcode.min.js",
  "/static/manifest.webmanifest",
  "/static/icons/icon.svg",
  "/static/icons/icon-192.png",
];

// Live, secret or one-shot: always straight to the network, never stored.
const BYPASS = [
  /^\/api(\/|$)/,
  /^\/socket\.io(\/|$)/,
  /^\/join(\/|$)/,
  /^\/sw\.js$/,
  /^\/tts\//,
  /^\/calibration\.png$/,
];

// Both reset when the browser restarts an idle worker, which then starts out network first again.
let lanSeen = false;
let retired = false;

function fromLanServer(response) {
  return response.headers.get(LAN_HEADER) === "1";
}

function cacheable(response) {
  // A redirect means the gate sent this device to /join: storing that would cache the PIN form as the page.
  if (!response || !response.ok || response.redirected || response.type !== "basic") return false;
  if (!fromLanServer(response)) return false;
  return !/no-store/i.test(response.headers.get("Cache-Control") || "");
}

async function deleteCaches(keep) {
  const names = await caches.keys();
  await Promise.all(names.filter((n) => n.startsWith("sar-") && n !== keep).map((n) => caches.delete(n)));
}

async function retire() {
  if (retired) return;
  retired = true;
  lanSeen = false;
  await deleteCaches(null);
  try {
    await self.registration.unregister();
  } catch {
    // already unregistered
  }
}

// Every network answer from this origin passes here: a LAN-mode one may be stored, an unmarked one retires us.
async function settle(request, response) {
  if (retired || response.type !== "basic") return;
  if (!fromLanServer(response)) {
    await retire();
    return;
  }
  lanSeen = true;
  if (!cacheable(response)) return;
  const copy = response.clone(); // before the first await, while the page has not read the body yet
  await (await caches.open(CACHE)).put(request, copy);
}

async function cached(request) {
  // caches.open() would create the cache again after retire() deleted it.
  if (retired || !(await caches.has(CACHE))) return undefined;
  return (await caches.open(CACHE)).match(request);
}

function fetchAndSettle(event) {
  const network = fetch(event.request);
  event.waitUntil(network.then((response) => settle(event.request, response)).catch(() => undefined));
  return network;
}

function offline() {
  const body =
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' +
    "<title>Can't reach the laptop</title>" +
    '<body style="font-family:system-ui,sans-serif;background:#f8f4ea;color:#1d2b45;padding:24px;line-height:1.5">' +
    "<h1 style=\"font-size:1.3rem\">Can't reach the laptop</h1>" +
    "<p>Check that this device is on the same Wi-Fi as the laptop and that the assistant is running, then reload.</p>";
  return new Response(body, { status: 503, headers: { "Content-Type": "text/html; charset=utf-8" } });
}

async function navigation(event) {
  try {
    return await fetchAndSettle(event);
  } catch {
    return (await cached(event.request)) || offline();
  }
}

async function asset(event) {
  const hit = lanSeen ? await cached(event.request) : undefined;
  const network = fetchAndSettle(event);
  if (hit) {
    network.catch(() => undefined);
    return hit;
  }
  try {
    return await network;
  } catch (err) {
    const fallback = await cached(event.request);
    if (fallback) return fallback;
    throw err;
  }
}

function route(request) {
  if (retired || request.method !== "GET" || request.headers.has("range")) return null;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return null;
  if (BYPASS.some((pattern) => pattern.test(url.pathname))) return null;
  if (request.mode === "navigate") return navigation;
  if (url.pathname.startsWith("/static/")) return asset;
  return null;
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      // One file at a time and best effort: a renamed file must not stop the worker from installing.
      await Promise.all(
        SHELL.map(async (path) => {
          try {
            const response = await fetch(path, { credentials: "same-origin", cache: "no-cache" });
            await settle(path, response);
          } catch {
            // offline or gated: it is stored on first use instead
          }
        }),
      );
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      await deleteCaches(CACHE);
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const strategy = route(event.request);
  if (strategy) event.respondWith(strategy(event));
});
