#!/usr/bin/env python3
"""
Optional voice tools added after the base MCP server (web/news search, later: image, sound, Spotify).
Every feature is behind its own env flag and degrades to a one-line message: a failing extra must never break the agent.
All tools here are plain sync functions: the bridge runs them in a worker thread (asyncio.to_thread), so network I/O
never touches the asyncio event loop or a PortAudio callback.
Revert: delete this file and the register_extras() call in server.py (or set the feature flag to 0).
"""

import threading
import time

from config import (
    GOOGLE_API_KEY, WEB_SEARCH_ENABLED, SEARCH_MODEL, SEARCH_MAX_CHARS, SEARCH_TIMEOUT_S,
    IMAGE_GEN_ENABLED, IMAGE_MODEL, IMAGE_SHOW_S, IMAGE_TIMEOUT_S, IMAGES_DIR,
    SOUND_GEN_ENABLED, SOUND_MODEL, SOUND_TIMEOUT_S, SOUNDS_DIR, SOUND_AUTOPLAY, SOUND_PLAY_GAIN,
    SPOTIFY_ENABLED, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_MARKET,
    SPOTIFY_PLAY_ENABLED, LIBRESPOT_BIN, LIBRESPOT_CACHE_DIR, SPOTIFY_DEVICE_NAME,
    SPOTIFY_BACKEND, SOLOIST_BIN, SOLOIST_DATA_DIR, SOLOIST_CACHE_DIR, SOLOIST_API_KEY,
)
from pathlib import Path

_SPOTIFY_TOKEN = {"value": "", "exp": 0.0}
_SPOTIFY_LOCK = threading.Lock()
_SOUND_BUSY = threading.Lock()
_IMAGE_BUSY = threading.Lock()  # one image at a time (protects the daily quota and the Pi)


def _web_search(query: str, news: bool) -> str:
    """One separate generate_content call with Google Search grounding (not part of the Live session)."""
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(timeout=int(SEARCH_TIMEOUT_S * 1000)),
    )
    kind = "latest news headlines about" if news else "up-to-date facts about"
    prompt = (
        f"Search the web for {kind}: {query}\n"
        f"Answer in the same language as the question. Maximum 3 short sentences, at most {SEARCH_MAX_CHARS} "
        "characters, plain text only: no markdown, no URLs, no lists. Say the most important facts first "
        "and mention the source name and date when it matters. This is read aloud."
    )
    resp = client.models.generate_content(
        model=SEARCH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )
    text = " ".join((resp.text or "").split())
    return text[:SEARCH_MAX_CHARS] if text else ""


def _spotify_token(httpx) -> str:
    """Client-credentials token, cached in memory until shortly before it expires."""
    with _SPOTIFY_LOCK:
        if _SPOTIFY_TOKEN["value"] and time.time() < _SPOTIFY_TOKEN["exp"] - 60:
            return _SPOTIFY_TOKEN["value"]
        r = httpx.post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"},
                       auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET), timeout=8)
        r.raise_for_status()
        j = r.json()
        _SPOTIFY_TOKEN["value"], _SPOTIFY_TOKEN["exp"] = j["access_token"], time.time() + float(j.get("expires_in", 3600))
        return _SPOTIFY_TOKEN["value"]


def _spotify_search(query: str) -> list:
    """[{'title','artist','album','uri'}] for the best track matches (max 3)."""
    import httpx
    hdr = {"Authorization": f"Bearer {_spotify_token(httpx)}"}
    r = httpx.get("https://api.spotify.com/v1/search", headers=hdr, timeout=8,
                  params={"q": query, "type": "track", "limit": 3, "market": SPOTIFY_MARKET})
    if r.status_code == 401:   # token revoked/expired early: refresh once
        _SPOTIFY_TOKEN["exp"] = 0.0
        hdr = {"Authorization": f"Bearer {_spotify_token(httpx)}"}
        r = httpx.get("https://api.spotify.com/v1/search", headers=hdr, timeout=8,
                      params={"q": query, "type": "track", "limit": 3, "market": SPOTIFY_MARKET})
    r.raise_for_status()
    return [{"title": t["name"], "artist": ", ".join(a["name"] for a in t["artists"]),
             "album": t["album"]["name"], "uri": t["uri"]} for t in r.json()["tracks"]["items"]]


def register_extras(mcp, hardware_manager=None):
    """Registers the optional tools whose flag is on. Called once from create_mcp_server()."""

    if WEB_SEARCH_ENABLED and GOOGLE_API_KEY:
        @mcp.tool()
        def web_search(query: str, news: bool = False) -> str:
            """
            Searches the web (or the latest news when news=true) and returns a short summary to read aloud.
            Use SPARINGLY, ONLY for news, very recent events, changing facts (scores, prices, releases), an explicit
            'search/look it up' request, or a very complex or obscure question. NEVER for general knowledge, science,
            history, definitions or how things work: answer those yourself.
            query: the search topic in English or Filipino. news: true for news/headlines.
            """
            if hardware_manager:
                hardware_manager.on_thinking("Searching...")
            try:
                out = _web_search(query, bool(news))
            except Exception as e:  # quota, timeout, network: log one line, tell Berry to apologise briefly
                print(f"⚠️ [WEB SEARCH]: failed ({type(e).__name__}): {str(e)[:300]}", flush=True)
                return "Search is unavailable right now. Tell the user briefly that you could not look it up."
            return out or "The search returned nothing useful. Tell the user briefly."
        print("🔎 [EXTRAS]: web_search enabled.", flush=True)

    if IMAGE_GEN_ENABLED and GOOGLE_API_KEY:
        def _image_worker(prompt: str):
            """Background thread: generate, save to workspace/images/, ask the GUI to show it. Never raises."""
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GOOGLE_API_KEY,
                                      http_options=types.HttpOptions(timeout=int(IMAGE_TIMEOUT_S * 1000)))
                resp = client.models.generate_content(
                    model=IMAGE_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"],
                                                       image_config=types.ImageConfig(aspect_ratio="16:9")))
                part = next((p for p in (resp.candidates[0].content.parts or []) if p.inline_data), None)
                if part is None:
                    print("⚠️ [IMAGE GEN]: model returned no image.", flush=True)
                    return
                ext = {"image/png": "png", "image/jpeg": "jpg"}.get(part.inline_data.mime_type, "jpg")
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                path = IMAGES_DIR / f"img_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
                path.write_bytes(part.inline_data.data)
                print(f"🖼️ [IMAGE GEN]: saved {path.name}", flush=True)
                pub = getattr(hardware_manager, "publisher", None)
                if pub:
                    pub.publish_image(path, IMAGE_SHOW_S)
                try:  # same completion cue as delegated tasks
                    hardware_manager.alarm_timer.buzzer.beep(duration=0.25)
                except Exception:
                    pass
            except Exception as e:
                print(f"⚠️ [IMAGE GEN]: failed ({type(e).__name__}): {str(e)[:300]}", flush=True)
            finally:
                _IMAGE_BUSY.release()

        @mcp.tool()
        def generate_image(prompt: str) -> str:
            """
            Generates a picture with Nano Banana 2 Lite and shows it fullscreen on the display for about 15 seconds.
            Takes a few seconds and uses API quota: call ONLY after the user confirmed by voice.
            prompt: detailed English description of the image (subject, style, colours, mood).
            """
            if not _IMAGE_BUSY.acquire(blocking=False):
                return "An image is already being generated. Tell the user to wait a moment."
            if hardware_manager:
                hardware_manager.on_thinking("Image...")
            threading.Thread(target=_image_worker, args=(prompt.strip()[:600],), daemon=True, name="image-gen").start()
            return "Started. The image will appear on the screen in a few seconds. Tell the user briefly."
        print("🖼️ [EXTRAS]: generate_image enabled.", flush=True)

    if SOUND_GEN_ENABLED and GOOGLE_API_KEY:
        def _sound_worker(prompt: str):
            """Background thread: Lyria 3 Clip -> workspace/sounds/*.mp3 (about 29 s, 44.1 kHz stereo). Never plays it."""
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GOOGLE_API_KEY,
                                      http_options=types.HttpOptions(timeout=int(SOUND_TIMEOUT_S * 1000)))
                resp = client.models.generate_content(
                    model=SOUND_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["AUDIO", "TEXT"]))
                part = next((p for p in (resp.candidates[0].content.parts or []) if p.inline_data), None)
                if part is None:
                    print("⚠️ [SOUND GEN]: model returned no audio.", flush=True)
                    return
                ext = "wav" if "wav" in (part.inline_data.mime_type or "") else "mp3"
                SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
                path = SOUNDS_DIR / f"snd_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"
                path.write_bytes(part.inline_data.data)
                print(f"🎼 [SOUND GEN]: saved {path.name} ({len(part.inline_data.data) // 1024} KB)", flush=True)
                try:  # same completion cue as delegated tasks / images
                    hardware_manager.alarm_timer.buzzer.beep(duration=0.25)
                except Exception:
                    pass
                mp = getattr(hardware_manager, "music_player", None)
                if SOUND_AUTOPLAY and mp is not None and hasattr(mp, "play_oneshot"):
                    if not mp.play_oneshot(path, SOUND_PLAY_GAIN):   # HDMI not available: stays saved only
                        print("🎼 [SOUND GEN]: not played (HDMI speaker unavailable); file is saved.", flush=True)
            except Exception as e:
                print(f"⚠️ [SOUND GEN]: failed ({type(e).__name__}): {str(e)[:300]}", flush=True)
            finally:
                _SOUND_BUSY.release()

        @mcp.tool()
        def generate_sound(prompt: str) -> str:
            """
            Generates a music/sound clip with Lyria 3 Clip (about 30 seconds, takes about 15 seconds) and SAVES it to
            workspace/sounds/ and plays it automatically on the monitor speaker when ready. Uses API quota: call ONLY after the user confirmed by voice.
            prompt: detailed English description (instruments, mood, tempo, genre; say 'instrumental, no vocals' if wanted).
            """
            if not _SOUND_BUSY.acquire(blocking=False):
                return "A sound is already being generated. Tell the user to wait a moment."
            if hardware_manager:
                hardware_manager.on_thinking("Sound...")
            threading.Thread(target=_sound_worker, args=(prompt.strip()[:600],), daemon=True, name="sound-gen").start()
            return "Started. About 15 seconds; the clip is saved and plays by itself on the monitor speaker. Tell the user briefly."
        print("🎼 [EXTRAS]: generate_sound enabled.", flush=True)

    if SPOTIFY_ENABLED:
        @mcp.tool()
        def spotify_search(query: str) -> str:
            """
            Looks up a song on Spotify and returns the best matches (title, artist). Search only: it does not play anything (use spotify_play for that).
            Use when the user asks to find/look up a specific song, artist or album by name.
            query: title and artist, e.g. 'Never Gonna Give You Up Rick Astley'.
            """
            if hardware_manager:
                hardware_manager.on_thinking("Spotify...")
            try:
                items = _spotify_search(query.strip()[:150])
            except Exception as e:
                print(f"⚠️ [SPOTIFY]: search failed ({type(e).__name__}): {str(e)[:200]}", flush=True)
                return "Spotify search is unavailable right now. Tell the user briefly."
            if not items:
                return f"No Spotify match for '{query}'. Tell the user briefly."
            top = items[0]
            out = f"Best match: '{top['title']}' by {top['artist']} (album {top['album']})."
            if len(items) > 1:
                out += " Also: " + "; ".join(f"'{i['title']}' by {i['artist']}" for i in items[1:]) + "."
            return out + ("" if SPOTIFY_PLAY_ENABLED else " Playback is not available yet.")
        print("🟢 [EXTRAS]: spotify_search enabled.", flush=True)

    if SPOTIFY_ENABLED and SPOTIFY_PLAY_ENABLED:
        import threading as _th
        from hardware.spotify_source import SpotifySource
        from mcp_server import spotify_player as sp_web
        _sp = {"playing": False}

        def _attach():
            """(music_player, hdmi_output) once the HDMI speaker is alive; also attaches the librespot source (once)."""
            mp = getattr(hardware_manager, "music_player", None)
            hdmi = getattr(mp, "hdmi", None)
            if mp is None or hdmi is None or not hdmi.alive:
                return None
            if hdmi.spotify is None:
                logs = Path.home() / ".local" / "state" / "berry"
                if SPOTIFY_BACKEND == "soloist":
                    from hardware.spotify_source import SoloistSource
                    hdmi.spotify = SoloistSource(SOLOIST_BIN, SOLOIST_DATA_DIR, SOLOIST_CACHE_DIR, SPOTIFY_DEVICE_NAME,
                                                 SOLOIST_API_KEY, log_dir=logs)
                else:
                    hdmi.spotify = SpotifySource(LIBRESPOT_BIN, LIBRESPOT_CACHE_DIR, SPOTIFY_DEVICE_NAME,
                                                 log_path=logs / "librespot.log")
                mp.on_play_hook = _local_music_started
            return mp, hdmi

        def _bg_pause():
            try:
                dev = sp_web.find_device(SPOTIFY_DEVICE_NAME, 0)
                if dev:
                    sp_web.control("pause", dev)
            except Exception as e:
                print(f"⚠️ [SPOTIFY]: pause failed ({type(e).__name__}): {str(e)[:120]}", flush=True)

        def _local_music_started():
            """Local music and Spotify never play together: starting local music pauses Spotify (in the background)."""
            if _sp["playing"]:
                _sp["playing"] = False
                _th.Thread(target=_bg_pause, daemon=True, name="spotify-pause").start()

        def _prewarm():
            """Waits for the HDMI speaker, then starts librespot so the first request is fast. Never raises."""
            try:
                for _ in range(60):
                    att = _attach()
                    if att:
                        att[1].spotify.ensure_running()
                        return
                    time.sleep(2)
                print("🟢 [SPOTIFY]: HDMI speaker not available; Spotify playback disabled for this run.", flush=True)
            except Exception as e:
                print(f"⚠️ [SPOTIFY]: prewarm failed ({type(e).__name__}): {e}", flush=True)
        _th.Thread(target=_prewarm, daemon=True, name="spotify-prewarm").start()

        _UNAVAIL = ("Spotify playback is unavailable right now (it needs the monitor speaker and Spotify's player). "
                    "Tell the user briefly, and offer the local music instead.")

        @mcp.tool()
        def spotify_play(query: str) -> str:
            """
            Finds a song on Spotify and plays it on the monitor speaker (Premium). Stops local music first.
            query: title and artist, e.g. 'Never Gonna Give You Up Rick Astley'.
            """
            if hardware_manager:
                hardware_manager.on_thinking("Spotify...")
            try:
                att = _attach()
                if not att or not att[1].spotify.ensure_running():
                    return _UNAVAIL
                mp, _hdmi = att
                items = _spotify_search(query.strip()[:150])
                if not items:
                    return f"No Spotify match for '{query}'. Tell the user briefly."
                top = items[0]
                dev = sp_web.find_device(SPOTIFY_DEVICE_NAME, 10.0)
                if not dev:
                    return (f"Found '{top['title']}' by {top['artist']} but Spotify's player is not ready yet. "
                            "Tell the user to try again in a few seconds.")
                mp.pause()   # local music off (Spotify has priority anyway)
                src = att[1].spotify
                before = src.chunks_out
                if not sp_web.play_uri(top["uri"], dev):
                    return f"Found '{top['title']}' by {top['artist']} but could not start it. Tell the user briefly."
                for _ in range(24):   # up to ~6 s: did audio really arrive? (librespot can be refused the track's decryption key)
                    if src.chunks_out - before >= 5:
                        break
                    time.sleep(0.25)
                else:
                    _bg_pause()
                    print("⚠️ [SPOTIFY]: no audio received from librespot (see ~/.local/state/berry/soloist.log / librespot.log).", flush=True)
                    return (f"Found '{top['title']}' by {top['artist']} but Spotify would not stream it to this device. "
                            "Tell the user honestly that it could not be played, and offer the local music.")
                _sp["playing"] = True
                return f"Playing '{top['title']}' by {top['artist']} on Spotify. Tell the user in one short sentence."
            except Exception as e:
                print(f"⚠️ [SPOTIFY]: play failed ({type(e).__name__}): {str(e)[:200]}", flush=True)
                return _UNAVAIL

        @mcp.tool()
        def spotify_control(action: str) -> str:
            """
            Controls Spotify playback. action: 'pause', 'resume', 'next' or 'previous'.
            Use only when the user clearly refers to the Spotify song (pause it, skip it, go back, continue it).
            """
            try:
                att = _attach()
                if not att:
                    return _UNAVAIL
                mp, _hdmi = att
                dev = sp_web.find_device(SPOTIFY_DEVICE_NAME, 0)
                if not dev:
                    return "Spotify is not playing anything right now. Tell the user briefly."
                a = action.lower().strip()
                if a in ("resume", "play", "next", "previous", "prev"):
                    mp.pause()   # keep local music off while Spotify runs
                ok = sp_web.control(a, dev)
                if ok:
                    _sp["playing"] = a != "pause"
                    return f"Spotify: {a} done. Tell the user in a few words."
                return "That Spotify control did not work (nothing active?). Tell the user briefly."
            except Exception as e:
                print(f"⚠️ [SPOTIFY]: control failed ({type(e).__name__}): {str(e)[:200]}", flush=True)
                return _UNAVAIL
        print("🟢 [EXTRAS]: spotify_play / spotify_control enabled.", flush=True)
