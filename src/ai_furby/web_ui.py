"""Phone control page for the Furby. Runs inside the voice loop and shares its Bluetooth connection.

Open http://<pi-address>:<WEB_PORT>/ on a phone on the same Wi-Fi. Every change needs the access token (WEB_TOKEN),
sent in the X-Token header. Without a token the page does not start. Only the actions in mood_actions.py can be run.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ai_furby.mood_actions import ACTION_SECONDS, MOOD_ACTIONS

log = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 8192

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furby</title>
<style>
  :root { color-scheme: light dark; --gap: 16px; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: var(--gap); background: #fff; color: #111; }
  @media (prefers-color-scheme: dark) { body { background: #111; color: #eee; } }
  h1 { font-size: 1.4rem; margin: 0 0 12px; }
  h2 { font-size: 1rem; margin: 20px 0 8px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; }
  button { font-size: 1rem; padding: 12px 14px; border-radius: 8px; border: 1px solid #888; background: #eee; color: #111; min-width: 96px; }
  @media (prefers-color-scheme: dark) { button { background: #333; color: #eee; } }
  input { font-size: 1rem; padding: 10px; width: 100%; box-sizing: border-box; }
  #msg { min-height: 1.2em; margin-top: 12px; }
</style>
</head>
<body>
<h1>Furby</h1>
<label>Access token<input id="token" type="password" autocomplete="off"></label>

<h2>Eyes and sleep</h2>
<div class="row">
  <button data-post="/api/eyes?on=1">Eyes on</button>
  <button data-post="/api/eyes?on=0">Eyes off</button>
  <button data-post="/api/sleep">Sleep</button>
  <button data-post="/api/wake">Wake</button>
</div>

<h2>Antenna</h2>
<div class="row">
  <button data-post="/api/antenna?r=255&g=0&b=0">Red</button>
  <button data-post="/api/antenna?r=0&g=255&b=0">Green</button>
  <button data-post="/api/antenna?r=0&g=0&b=255">Blue</button>
  <button data-post="/api/antenna?r=255&g=160&b=0">Amber</button>
  <button data-post="/api/antenna?r=0&g=0&b=0">Off</button>
</div>

<h2>Moods</h2>
<div class="row">
  <button data-post="/api/mood?name=happy">Happy</button>
  <button data-post="/api/mood?name=excited">Excited</button>
  <button data-post="/api/mood?name=curious">Curious</button>
  <button data-post="/api/mood?name=sad">Sad</button>
  <button data-post="/api/mood?name=sleepy">Sleepy</button>
  <button data-post="/api/mood?name=silly">Silly</button>
</div>

<div id="msg"></div>
<script>
  const tokenBox = document.getElementById('token');
  const msg = document.getElementById('msg');
  try { tokenBox.value = localStorage.getItem('furbyToken') || ''; } catch (e) {}
  tokenBox.addEventListener('change', () => { try { localStorage.setItem('furbyToken', tokenBox.value); } catch (e) {} });
  document.querySelectorAll('button[data-post]').forEach(button => {
    button.addEventListener('click', async () => {
      msg.textContent = 'Sending...';
      try {
        const response = await fetch(button.dataset.post, { method: 'POST', headers: { 'X-Token': tokenBox.value } });
        msg.textContent = response.ok ? 'Done' : 'Error ' + response.status + ': ' + (await response.text());
      } catch (e) {
        msg.textContent = 'Could not reach the Furby page: ' + e;
      }
    });
  });
</script>
</body>
</html>
"""


async def handle(method: str, target: str, headers: dict[str, str], furby: Any, token: str) -> tuple[int, str, str]:
    """Returns (status, content type, body) for one request. Kept free of sockets so it can be tested."""
    parts = urlsplit(target)
    if method == "GET" and parts.path == "/":
        return 200, "text/html; charset=utf-8", PAGE
    if method != "POST" or not parts.path.startswith("/api/"):
        return 404, "text/plain; charset=utf-8", "Not found"
    if not token or headers.get("x-token", "") != token:
        return 401, "text/plain; charset=utf-8", "Wrong or missing token"

    query = {key: values[0] for key, values in parse_qs(parts.query).items()}
    route = parts.path
    try:
        if route == "/api/eyes":
            await furby.set_backlight(query.get("on") == "1")
        elif route == "/api/wake":
            await furby.set_backlight(True)
        elif route == "/api/sleep":
            await furby.set_antenna_color(0, 0, 0)
            await furby.set_backlight(False)
        elif route == "/api/antenna":
            red, green, blue = (_channel(query, name) for name in ("r", "g", "b"))
            await furby.set_antenna_color(red, green, blue)
        elif route == "/api/mood":
            action = MOOD_ACTIONS.get(query.get("name", ""))
            if action is None:
                return 400, "text/plain; charset=utf-8", "Unknown mood"
            length = ACTION_SECONDS.get(action, 2.5)
            await furby.act_for(action, seconds=length, repeat_seconds=length)
        else:
            return 404, "text/plain; charset=utf-8", "Not found"
    except ValueError as error:
        return 400, "text/plain; charset=utf-8", str(error)
    return 200, "text/plain; charset=utf-8", "ok"


def _channel(query: dict[str, str], name: str) -> int:
    try:
        value = int(query.get(name, ""))
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number from 0 to 255") from error
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be from 0 to 255")
    return value


async def serve(furby: Any, port: int, token: str, host: str = "0.0.0.0") -> None:
    """Listens for phone requests until cancelled."""

    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
            if len(raw) > MAX_REQUEST_BYTES:
                status, ctype, body = 400, "text/plain; charset=utf-8", "Request too large"
            else:
                lines = raw.decode("latin-1").split("\r\n")
                method, target, _version = lines[0].split(" ", 2)
                headers = {}
                for line in lines[1:]:
                    if ":" in line:
                        name, value = line.split(":", 1)
                        headers[name.strip().lower()] = value.strip()
                status, ctype, body = await handle(method, target, headers, furby, token)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ValueError):
            status, ctype, body = 400, "text/plain; charset=utf-8", "Bad request"
        except Exception:
            log.warning("Phone request failed", exc_info=True)
            status, ctype, body = 500, "text/plain; charset=utf-8", "Furby command failed"
        payload = body.encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 500: "Server Error"}.get(status, "")
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n"
            "Connection: close\r\n\r\n".encode("latin-1") + payload
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(connection, host, port)
    log.info("Phone page listening on port %d", port)
    async with server:
        await server.serve_forever()
