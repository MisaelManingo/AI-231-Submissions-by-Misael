#!/usr/bin/env python3
"""
Spotify Web API playback control (needs Premium) for the librespot Connect device that hardware/spotify_source.py runs.
User token: obtained ONCE with `python -m mcp_server.spotify_player auth` (opens an authorize URL, catches the redirect on
127.0.0.1) and stored OUTSIDE the repo in ~/.local/share/berry-librespot/web_token.json (mode 600). Refreshed automatically.
All functions are plain blocking calls meant for tool worker threads (never the asyncio loop).
"""

import sys
import json
import time
import secrets
import threading
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_TOKEN_FILE, SPOTIFY_REDIRECT_URI

SCOPES = "user-read-playback-state user-modify-playback-state"
_LOCK = threading.Lock()
_CACHE = {"token": "", "exp": 0.0}


def _save(tok: dict):
    p = Path(SPOTIFY_TOKEN_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(tok))
    tmp.chmod(0o600)
    tmp.replace(p)


def _token_request(data: dict) -> dict:
    import httpx
    r = httpx.post("https://accounts.spotify.com/api/token", data=data,
                   auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET), timeout=10)
    r.raise_for_status()
    return r.json()


def user_token() -> str:
    with _LOCK:
        if _CACHE["token"] and time.time() < _CACHE["exp"] - 60:
            return _CACHE["token"]
        stored = json.loads(Path(SPOTIFY_TOKEN_FILE).read_text())
        j = _token_request({"grant_type": "refresh_token", "refresh_token": stored["refresh_token"]})
        if j.get("refresh_token"):   # Spotify may rotate it
            stored["refresh_token"] = j["refresh_token"]
            _save(stored)
        _CACHE["token"], _CACHE["exp"] = j["access_token"], time.time() + float(j.get("expires_in", 3600))
        return _CACHE["token"]


def _api(method: str, path: str, **kw):
    import httpx
    for attempt in (0, 1):
        r = httpx.request(method, "https://api.spotify.com/v1" + path, timeout=8,
                          headers={"Authorization": f"Bearer {user_token()}"}, **kw)
        if r.status_code == 401 and attempt == 0:
            _CACHE["exp"] = 0.0
            continue
        return r
    return r


def find_device(name: str, wait_s: float = 8.0):
    """Id of the Connect device called `name` (polls while librespot is still registering), else None."""
    end = time.time() + wait_s
    while True:
        r = _api("GET", "/me/player/devices")
        if r.status_code == 200:
            for d in r.json().get("devices", []):
                if d.get("name") == name:
                    return d["id"]
        if time.time() >= end:
            return None
        time.sleep(1.0)


def play_uri(uri: str, device_id: str) -> bool:
    r = _api("PUT", "/me/player/play", params={"device_id": device_id}, json={"uris": [uri]})
    if r.status_code not in (200, 202, 204):
        print(f"⚠️ [SPOTIFY]: play failed HTTP {r.status_code}: {r.text[:160]}", flush=True)
    return r.status_code in (200, 202, 204)


def control(action: str, device_id: str) -> bool:
    """action: pause | resume | next | previous"""
    a = action.lower().strip()
    if a == "pause":
        r = _api("PUT", "/me/player/pause", params={"device_id": device_id})
    elif a in ("resume", "play"):
        r = _api("PUT", "/me/player/play", params={"device_id": device_id})
    elif a == "next":
        r = _api("POST", "/me/player/next", params={"device_id": device_id})
    elif a in ("previous", "prev"):
        r = _api("POST", "/me/player/previous", params={"device_id": device_id})
    else:
        return False
    if r.status_code not in (200, 202, 204):
        print(f"⚠️ [SPOTIFY]: {a} failed HTTP {r.status_code}: {r.text[:160]}", flush=True)
    return r.status_code in (200, 202, 204)


def _auth_cli():
    """One-time authorization. Open the printed URL in the browser ON THIS Pi: the redirect is caught on 127.0.0.1."""
    import http.server
    state = secrets.token_urlsafe(12)
    u = urlparse(SPOTIFY_REDIRECT_URI)
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = parse_qs(urlparse(self.path).query)
            if urlparse(self.path).path == u.path and q.get("state", [""])[0] == state and "code" in q:
                got["code"] = q["code"][0]
                msg = b"Berry is authorized for Spotify. You can close this tab."
            else:
                msg = b"Not the expected callback."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer((u.hostname, u.port), H)
    srv.timeout = 1.0
    url = "https://accounts.spotify.com/authorize?" + urlencode({
        "client_id": SPOTIFY_CLIENT_ID, "response_type": "code", "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SCOPES, "state": state})
    print("OPEN THIS URL in the browser on the Pi and approve:\n" + url, flush=True)
    end = time.time() + 1800
    while "code" not in got and time.time() < end:
        srv.handle_request()
    srv.server_close()
    if "code" not in got:
        print("Timed out waiting for authorization.", flush=True)
        sys.exit(1)
    j = _token_request({"grant_type": "authorization_code", "code": got["code"], "redirect_uri": SPOTIFY_REDIRECT_URI})
    _save({"refresh_token": j["refresh_token"], "scope": j.get("scope", "")})
    print("Authorized. Token saved to", SPOTIFY_TOKEN_FILE, "(scope:", j.get("scope"), ")", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        _auth_cli()
    else:
        print("usage: python -m mcp_server.spotify_player auth")
