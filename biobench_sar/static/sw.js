// Service worker for the dashboard: quick reloads from a cached shell, never in the way of live data.
//   page loads (navigations)  network first; the cached page when the laptop is slow or unreachable
//   /static/*                 stale-while-revalidate: the cached copy now, a fresh one stored for next time
//   /api/*                    network first; a stored JSON read only when the laptop can't be reached
//   everything else           untouched: Socket.IO, POSTs, other sites, camera frames, speech audio, /join
// Serve it from /sw.js (scope "/"), not /static/sw.js, so it can cover /dashboard. Browsers run service
// workers only on https:// or localhost, so phones on plain-HTTP Wi-Fi skip it; pages must work without it.
"use strict";

const CACHE_VERSION = "v1";
const CACHE = `sar-shell-${CACHE_VERSION}`;
const NAVIGATION_TIMEOUT_MS = 4000;

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

// Live, large, one-shot or secret: never answered from (or stored in) the cache.
const BYPASS = [
  /^\/socket\.io\//,
  /^\/join(\/|$)/,
  /^\/sw\.js$/,
  /^\/api\/frame\.jpg$/,
  /^\/api\/listen$/,
  /^\/tts\//,
  /^\/calibration\.png$/,
];

function cacheable(response) {
  // A redirect means the gate sent this device to /join: storing that would cache the PIN form as the page.
  if (!response || !response.ok || response.redirected || response.type !== "basic") return false;
  const control = response.headers.get("Cache-Control") || "";
  return !/no-store/i.test(control);
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

function fromCache(response) {
  // Marks a stand-in answer, so a page can tell stale data from live data.
  const headers = new Headers(response.headers);
  headers.set("X-SAR-From-Cache", "1");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function fetchAndStore(event, cache) {
  const network = fetch(event.request);
  event.waitUntil(
    network
      .then((response) => (cacheable(response) ? cache.put(event.request, response.clone()) : undefined))
      .catch(() => undefined),
  );
  return network;
}

function timeout(ms) {
  return new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms));
}

async function navigation(event) {
  const cache = await caches.open(CACHE);
  const network = fetchAndStore(event, cache);
  try {
    return await Promise.race([network, timeout(NAVIGATION_TIMEOUT_MS)]);
  } catch {
    const cached = await cache.match(event.request);
    if (cached) return cached;
    return network.catch(offline);
  }
}

async function asset(event) {
  const cache = await caches.open(CACHE);
  const cached = await cache.match(event.request);
  const network = fetchAndStore(event, cache);
  if (cached) {
    network.catch(() => undefined);
    return cached;
  }
  return network;
}

async function api(event) {
  const cache = await caches.open(CACHE);
  const url = new URL(event.request.url);
  try {
    const response = await fetch(event.request);
    // Only plain JSON reads without a query: a bounded set, so the cache can't grow with every search.
    const json = (response.headers.get("Content-Type") || "").includes("application/json");
    if (!url.search && json && cacheable(response)) event.waitUntil(cache.put(event.request, response.clone()));
    return response;
  } catch (err) {
    const cached = await cache.match(event.request);
    if (cached) return fromCache(cached);
    throw err;
  }
}

function route(request) {
  if (request.method !== "GET" || request.headers.has("range")) return null;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return null;
  if (BYPASS.some((pattern) => pattern.test(url.pathname))) return null;
  if (request.mode === "navigate") return navigation;
  if (url.pathname.startsWith("/api/")) return api;
  if (url.pathname.startsWith("/static/")) return asset;
  return null;
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // One file at a time and best effort: a renamed file must not stop the worker from installing.
      await Promise.all(
        SHELL.map(async (path) => {
          try {
            const response = await fetch(path, { credentials: "same-origin", cache: "no-cache" });
            if (cacheable(response)) await cache.put(path, response);
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
      const names = await caches.keys();
      await Promise.all(names.filter((n) => n.startsWith("sar-") && n !== CACHE).map((n) => caches.delete(n)));
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const strategy = route(event.request);
  if (strategy) event.respondWith(strategy(event));
});
