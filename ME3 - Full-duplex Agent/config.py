#!/usr/bin/env python3
"""
Configuration for ME3 Full-Duplex Voice Assistant (Raspberry Pi 5)
Centralized configuration for:
- Gemini Live model, voice, system instruction
- Audio I/O settings (G-MARK Micro Go USB)
- Hardware GPIO pins and I2C addresses
- Virtual thermostat & Open-Meteo weather coordinates
- Telephony & state persistence paths
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for ME3 project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# -----------------------------------------------------------------------------
# GEMINI LIVE MODEL & AGENT SETTINGS
# -----------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.8-live")
GEMINI_VOICE_ID = os.getenv("GEMINI_VOICE_ID", "Kore")  # Aoede, Kore, Zephyr, Puck, Charon, Fenrir
GEMINI_LANGUAGE = os.getenv("GEMINI_LANGUAGE", "en")
DELEGATE_AGENT_MODEL = os.getenv("DELEGATE_AGENT_MODEL", "gemini-3.8-flash")
DELEGATE_THINKING_LEVEL = os.getenv("DELEGATE_THINKING_LEVEL", "LOW")  # LOW or MEDIUM for Antigravity
DELEGATE_THINKING_BUDGET = int(os.getenv("DELEGATE_THINKING_BUDGET", "1024"))  # Low/Medium thinking budget (tokens)

# Attention gate: the agent only answers speech directed at it (name "Berry", ongoing conversation, or a clear
# command/question to an assistant); anything else makes the model call ignore_speech and the reply is dropped locally.
# ME3_ATTENTION=0 disables it (agent answers everything it hears, as before).
ATTENTION_ENABLED = os.getenv("ME3_ATTENTION", "1").lower() in ("1", "true", "yes")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Berry")


SYSTEM_INSTRUCTION = (
    "You are an intelligent, responsive, full-duplex AI voice assistant running on a Raspberry Pi 5. "
    "You communicate via real-time speech. Speak naturally, concisely, and warmly. "
    "Keep spoken responses brief (1 to 2 sentences) because the user is listening via voice in real-time. "
    "CRITICAL LANGUAGE RULE: You must strictly understand and speak in English at all times. Never switch to another language regardless of accent or acoustic noise. "
    "You have full access to local Raspberry Pi smart-home actions and general agent tools via function calls. "
    "When a user asks to turn on/off or change lights, play/pause music, set timers or alarms, "
    "check the weather or time, adjust the virtual thermostat, save reminders, make calls or messages, "
    "or execute general agent tasks, execute the matching tool. "
    "LONG TASKS: whenever the user asks you to program, code, build, make or create something that takes real work - a game "
    "(e.g. flappy bird), a web app or website, a script, a program, a multi-file project, or a written report - you MUST call "
    "delegate_long_task straight away. Put a complete, detailed specification in task_instructions (what to build, features, "
    "controls, look and behaviour, edge cases) and a short task_name. Then say ONE short sentence like 'On it, I\'m building it in the "
    "background.' and keep chatting normally. NEVER write code, HTML or long text yourself, and do not use create_simple_website "
    "for apps or games (only for a trivial one-paragraph page). The finished files appear in the workspace folder; if asked about "
    "progress or where the files are, call get_background_task_status. "
    + ("ATTENTION RULE (highest priority): Your name is Berry. Respond ONLY when the speech is directed TO YOU. That is the case when: "
        "(a) the user addresses you by name, usually at the start ('Berry, ...', 'Hey Berry ...'); "
        "(b) it is clearly a follow-up in a conversation you are already having with the user (you spoke to them in the last ~20 seconds); or "
        "(c) it is clearly a request, question or command meant for an assistant ('turn on the lights', 'what is the weather', 'play music'). "
        "If the speech is NOT for you (people talking to each other, someone talking to themselves, a phone call, TV, music, background chatter, "
        "a sentence addressed to someone else by name), you must not reply at all: immediately call the ignore_speech tool and produce NO speech. "
        "When in doubt about whether you are being addressed, call ignore_speech. Never call ignore_speech when the user said 'Berry'. "
      if ATTENTION_ENABLED else "") +
    "CRITICAL: After executing any tool EXCEPT ignore_speech, you MUST immediately speak a concise verbal confirmation to the user "
    "(for example, 'I turned on the lights', 'The weather in Manila is 28 degrees and partly cloudy'). "
    "Never remain silent after a tool call (the only exception is ignore_speech, after which you stay completely silent). "
    "CRITICAL TOOL CALLING RULE: ONLY call a tool if the user explicitly requested that exact action (ignore_speech is the exception, per the ATTENTION RULE). "
    "NEVER invoke zero-argument tools like cancel_alarm, cancel_timer, or get_time_date unless the user specifically and clearly said 'cancel alarm', 'stop timer', or asked for the time. "
    "NEVER execute bash commands (execute_bash_command) unless the user explicitly said 'run bash command' or 'run shell command'. Under NO circumstances should you ever reboot, restart, or shut down the device. "
    "If the user says 'play music', 'play song', 'play track', or mentions music, ALWAYS invoke the 'control_media' tool with action='play'. NEVER use bash or shell for music. "
    "If the user speech is conversational, ambiguous, tells a joke, or asks a general question, respond conversationally or ask for clarification—DO NOT invoke tools at random. "
    "Note on thermostat: you control a virtual software thermostat; there is no internal room sensor, "
    "so you report your software setpoint, mode, and compare it with the current live outdoor temperature. "
    "Note on workspace & files: you have full permission to create, read, and modify user files inside your dedicated 'workspace/' folder."
)

# BILINGUAL MODE: default "en,fil" (English + Filipino/Taglish, verified working). Set ME3_LANGS=en for the original English-only lock.
# Bilingual lifts three English locks: the prompt rule, the speech_config.language_code, and the en-US-only transcription.
LANGS = [l.strip().lower() for l in os.getenv("ME3_LANGS", "en,fil").split(",") if l.strip()]
BILINGUAL = "fil" in LANGS or "tl" in LANGS
_EN_RULE = ("CRITICAL LANGUAGE RULE: You must strictly understand and speak in English at all times. "
            "Never switch to another language regardless of accent or acoustic noise. ")
_BI_RULE = ("LANGUAGE RULE: The user may speak English, Filipino (Tagalog), or a mix of both (Taglish); understand all of them. "
            "Always reply in the language the user last used: English for English, Filipino for Filipino, and for Taglish "
            "mix naturally while mostly matching them. Never use any other language and never switch languages on your own. "
            "RESPOND UNMISTAKABLY IN THE USER'S LANGUAGE. Tool calls and their arguments always stay in English. ")
if BILINGUAL:
    assert _EN_RULE in SYSTEM_INSTRUCTION
    SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION.replace(_EN_RULE, _BI_RULE)

# EMPATHY: Gemini 3.8 Live has affective dialog on by default (no API switch); this only tells the model to use it
# on purpose. ME3_EMPATHY=0 removes the sentence.
if os.getenv("ME3_EMPATHY", "1") != "0":
    SYSTEM_INSTRUCTION += (
        " EMOTIONAL AWARENESS: Listen to the user's tone of voice (stress, excitement, sadness, frustration, hurry, "
        "tiredness) and respond accordingly: match their energy, soften and slow down when they sound upset, be upbeat "
        "when they are happy, and briefly acknowledge strong feelings in one short phrase. Stay concise.")

# WEB / NEWS SEARCH (mcp_server/extras.py): Gemini Google-Search grounding in a separate call. ME3_WEB_SEARCH=0 disables.
WEB_SEARCH_ENABLED = os.getenv("ME3_WEB_SEARCH", "1").lower() in ("1", "true", "yes")
SEARCH_MODEL = os.getenv("SEARCH_MODEL", "gemini-3.8-flash")
SEARCH_MAX_CHARS = int(os.getenv("SEARCH_MAX_CHARS", "400"))
SEARCH_TIMEOUT_S = max(10.0, float(os.getenv("SEARCH_TIMEOUT_S", "25")))  # API minimum deadline is 10 s
if WEB_SEARCH_ENABLED and GOOGLE_API_KEY:
    SYSTEM_INSTRUCTION += (
        " WEB SEARCH (use sparingly): answer from your own knowledge by default, including general explanations, science, "
        "history, how things work, definitions, advice and chat (e.g. 'how does a laser work'). Call web_search ONLY when the "
        "answer needs information after your knowledge cutoff or that changes (news, recent events, live scores, prices, "
        "election results, new releases), or the user explicitly says to search/look it up, or it is a very complex or obscure "
        "factual question you cannot answer reliably. Never search for common knowledge. When you do, first say a very short "
        "'let me check' in the user's language, then read the short result aloud in the user's language.")

# IMAGE GENERATION (mcp_server/extras.py): Nano Banana 2 Lite in a background thread, shown as a GUI overlay.
# ME3_IMAGE_GEN=0 disables. Files go to workspace/images/ (local only, never uploaded).
IMAGE_GEN_ENABLED = os.getenv("ME3_IMAGE_GEN", "1").lower() in ("1", "true", "yes")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-lite-image")
IMAGE_SHOW_S = float(os.getenv("IMAGE_SHOW_S", "15"))
IMAGE_TIMEOUT_S = max(10.0, float(os.getenv("IMAGE_TIMEOUT_S", "60")))
IMAGES_DIR = BASE_DIR / "workspace" / "images"
if IMAGE_GEN_ENABLED and GOOGLE_API_KEY:
    SYSTEM_INSTRUCTION += (
        " IMAGE GENERATION: when the user asks you to draw, make or generate a picture, first say what you will create in one "
        "short sentence and ask 'okay?' in the user's language; only after they say yes, call generate_image with a detailed "
        "English prompt, then say in one short sentence that it is being made and will appear on the screen.")

# SOUND-BYTE GENERATION (mcp_server/extras.py): Lyria 3 Clip in a background thread, saved to workspace/sounds/.
# Generation + saving only: NOTHING is played by this feature. ME3_SOUND_GEN=0 disables.
SOUND_GEN_ENABLED = os.getenv("ME3_SOUND_GEN", "1").lower() in ("1", "true", "yes")
SOUND_MODEL = os.getenv("SOUND_MODEL", "lyria-3-clip-preview")
SOUND_TIMEOUT_S = max(10.0, float(os.getenv("SOUND_TIMEOUT_S", "90")))
SOUNDS_DIR = BASE_DIR / "workspace" / "sounds"
# Generated clips play automatically on the HDMI speaker via MusicPlayer.play_oneshot (never on the USB speaker), at most
# the default music volume x SOUND_PLAY_GAIN. SOUND_AUTOPLAY=0 = save only.
SOUND_AUTOPLAY = os.getenv("SOUND_AUTOPLAY", "1").lower() in ("1", "true", "yes")
SOUND_PLAY_GAIN = float(os.getenv("SOUND_PLAY_GAIN", "0.8"))
if SOUND_GEN_ENABLED and GOOGLE_API_KEY:
    SYSTEM_INSTRUCTION += (
        " SOUND GENERATION: when the user asks for a sound effect, jingle, tune or short music clip, first say what you will "
        "create in one short sentence and ask 'okay?' in the user's language; only after they say yes, call generate_sound "
        "with a detailed English prompt, then say in one short sentence that it takes about 15 seconds and will play "
        "by itself on the monitor speaker when ready.")

# SPOTIFY SEARCH (mcp_server/extras.py): Web API client-credentials search only (previews are gone for new apps, so this
# finds songs; playback is a separate feature). Needs SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in .env. ME3_SPOTIFY=0 disables.
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_ENABLED = os.getenv("ME3_SPOTIFY", "1").lower() in ("1", "true", "yes") and bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)
SPOTIFY_MARKET = os.getenv("SPOTIFY_MARKET", "PH")
# SPOTIFY PLAYBACK: librespot (official Raspotify build, unpacked in ~/.local/share/berry-librespot, no system install) is a
# Connect device whose audio is piped into the HDMI music writer thread (hardware/spotify_source.py). Needs Premium, a one-time
# librespot login (cache dir) and a one-time Web API authorization (token file). Anything missing => search-only, as before.
# ME3_SPOTIFY_PLAY=0 disables playback.
_SP_DIR = Path.home() / ".local" / "share" / "berry-librespot"
LIBRESPOT_BIN = Path(os.getenv("LIBRESPOT_BIN", str(_SP_DIR / "librespot")))
LIBRESPOT_CACHE_DIR = Path(os.getenv("LIBRESPOT_CACHE_DIR", str(_SP_DIR / "cache")))
SPOTIFY_TOKEN_FILE = Path(os.getenv("SPOTIFY_TOKEN_FILE", str(_SP_DIR / "web_token.json")))
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SPOTIFY_DEVICE_NAME = os.getenv("SPOTIFY_DEVICE_NAME", "Berry")
# Backend 1 (preferred): Spotify Soloist, Spotify's official headless client (binary in ~/.local/share/berry-soloist, never
# committed; needs SOLOIST_API_KEY in .env and a one-time pairing, marked by the file `paired`). librespot is the fallback.
_SO_DIR = Path.home() / ".local" / "share" / "berry-soloist"
SOLOIST_BIN = Path(os.getenv("SOLOIST_BIN", str(_SO_DIR / "soloist")))
SOLOIST_DATA_DIR = _SO_DIR / "data"
SOLOIST_CACHE_DIR = _SO_DIR / "cache"
SOLOIST_API_KEY = os.getenv("SOLOIST_API_KEY", "")
_SOLOIST_READY = SOLOIST_BIN.exists() and bool(SOLOIST_API_KEY) and (_SO_DIR / "paired").exists()
_LIBRESPOT_READY = (LIBRESPOT_BIN.exists() and SPOTIFY_TOKEN_FILE.exists()
                    and (LIBRESPOT_CACHE_DIR / "credentials.json").exists())
SPOTIFY_BACKEND = "soloist" if _SOLOIST_READY else "librespot"
SPOTIFY_PLAY_ENABLED = (SPOTIFY_ENABLED and os.getenv("ME3_SPOTIFY_PLAY", "1").lower() in ("1", "true", "yes")
                        and SPOTIFY_TOKEN_FILE.exists() and (_SOLOIST_READY or _LIBRESPOT_READY))
if SPOTIFY_ENABLED:
    if SPOTIFY_PLAY_ENABLED:
        SYSTEM_INSTRUCTION += (
            " SPOTIFY: to play a specific song or artist on Spotify, call spotify_play with the title and artist, then say in "
            "one short sentence what is playing. To pause, resume, skip or go back on Spotify call spotify_control. "
            "spotify_search only looks a song up. Spotify and the local music never play together. A plain 'play music' "
            "request (no song named, or 'my music') still uses control_media; 'stop music' / 'pause' while Spotify plays "
            "should use spotify_control. Playback needs the monitor speaker: if a tool says it is unavailable, tell the user.")
    else:
        SYSTEM_INSTRUCTION += (
            " SPOTIFY: when the user asks you to find or look up a specific song, artist or album by name, call spotify_search "
            "with the title and artist, then tell them in one short sentence what you found. You cannot play Spotify songs yet: "
            "say so and offer the local music instead. A plain 'play music' request still uses control_media.")

if not ATTENTION_ENABLED:  # no ignore_speech tool is registered: drop every mention of it
    SYSTEM_INSTRUCTION = (SYSTEM_INSTRUCTION.replace(" EXCEPT ignore_speech", "")
                          .replace(" (the only exception is ignore_speech, after which you stay completely silent)", "")
                          .replace(" (ignore_speech is the exception, per the ATTENTION RULE)", ""))

# -----------------------------------------------------------------------------
# AUDIO I/O & VAD CONFIGURATION
# -----------------------------------------------------------------------------
# Search substring for G-MARK Micro Go USB audio device
AUDIO_DEVICE_NAME = os.getenv("AUDIO_DEVICE_NAME", "USB Composite Device")
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", None)    # Device ID or name substring
AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", None)  # Device ID or name substring

# Audio Hardware & Model Sample Rates
AUDIO_HARDWARE_RATE = 48000   # Native rate required by G-MARK Micro Go ALSA driver
AUDIO_MODEL_RATE = 16000      # 16 kHz mono for Silero VAD & Gemini Live
AUDIO_DECIMATION_FACTOR = 3   # 48000 / 16000 = exactly 3
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1            # Mono
AUDIO_CHUNK_SIZE = 512        # 32ms frames @ 16 kHz (1536 frames @ 48 kHz)

# Silero VAD parameters
VAD_CONFIDENCE_THRESHOLD = 0.45   # Calibrated: captures unstressed words ('a', 'the', 'in', 'me', 'to')
VAD_START_SPEECH_MS = 100        # ~100ms to confirm speech onset
VAD_STOP_SPEECH_MS = 750         # 750ms max silence to mark turn end (prevents cutting off multi-word questions like 'What is the weather in Manila?')

# Local Voice Barge-In parameters (over speaker playback)
BARGE_IN_CONFIDENCE_THRESHOLD = float(os.getenv("BARGE_IN_CONFIDENCE_THRESHOLD", "0.85"))
BARGE_IN_RMS_THRESHOLD = float(os.getenv("BARGE_IN_RMS_THRESHOLD", "0.40"))

# Software echo cancellation (agent/aec.py). OFF by default: measured on the G-MARK (shared housing) it never
# converges (mic saturates at agent volume + ~300-400 ppm mic/speaker clock drift). Superseded by listen-probes below.
# ME3_AEC=1 turns it on again.
AEC_ENABLED = os.getenv("ME3_AEC", "0").lower() in ("1", "true", "yes")
AEC_PARTITIONS = int(os.getenv("AEC_PARTITIONS", "8"))                 # filter span = N * 32 ms
AEC_MU = float(os.getenv("AEC_MU", "0.6"))                             # adaptation speed
AEC_CONVERGED_ERLE_DB = float(os.getenv("AEC_CONVERGED_ERLE_DB", "6")) # below this, fall back to legacy rule
AEC_BARGE_CONFIDENCE = float(os.getenv("AEC_BARGE_CONFIDENCE", "0.60"))  # Silero conf on echo-cancelled audio
AEC_BARGE_MIN_RMS = float(os.getenv("AEC_BARGE_MIN_RMS", "0.02"))        # residual RMS floor
AEC_BARGE_FLOOR_MULT = float(os.getenv("AEC_BARGE_FLOOR_MULT", "3.0"))   # x learned residual-echo level

# LISTEN-PROBE barge-in (agent/vad.py + audio_transport.py). Hardware facts measured on the G-MARK:
# the mic hears the speaker ~+24 dB hot (saturates at agent volume) with a stable ~161 ms speaker->mic delay.
# So instead of cancelling echo we briefly PAUSE the agent's playback (queued speech is kept, nothing is lost);
# ~ECHO_DELAY later the mic is echo-free and any loud speech there is the user. Natural pauses in the agent's
# speech count as listening windows too, so probes only fire when no echo-free window occurred recently.
# ME3_DUCK_PROBE=0 disables it (falls back to AEC / legacy raw-mic rule).
DUCK_PROBE_ENABLED = os.getenv("ME3_DUCK_PROBE", "1").lower() in ("1", "true", "yes")
PROBE_INTERVAL_MS = int(os.getenv("PROBE_INTERVAL_MS", "1750"))   # max time agent talks without a listening window
PROBE_PAUSE_MS = int(os.getenv("PROBE_PAUSE_MS", "340"))          # pause length of one probe (echo-free part ~ this - 200 ms)
PROBE_HOLD_MAX_MS = int(os.getenv("PROBE_HOLD_MAX_MS", "2000"))   # never keep the agent paused longer than this
ECHO_DELAY_MS = float(os.getenv("ECHO_DELAY_MS", "161"))          # measured speaker->mic delay (see tests/hw notes)
ECHO_COUPLING = float(os.getenv("ECHO_COUPLING", "16"))           # mic rms per unit speaker rms (linear region)
PROBE_ECHO_OK = float(os.getenv("PROBE_ECHO_OK", "0.04"))         # frame counts as echo-free if predicted echo rms <= this
PROBE_MIN_RMS = float(os.getenv("PROBE_MIN_RMS", "0.08"))         # user speech must exceed this mic rms in a listening window
# Pause at a word boundary: when a probe is due, wait (up to PROBE_DEFER_MS) until the next queued audio is quiet.
PROBE_QUIET_RMS = float(os.getenv("PROBE_QUIET_RMS", "0.012"))    # next ~80 ms of queued speaker audio below this = boundary
PROBE_DEFER_MS = int(os.getenv("PROBE_DEFER_MS", "500"))

# Music loudness (hardware/music_player.py). The mic shares a housing with the speaker, so music masks the user's
# voice commands. Measured with Gemini transcription over real mic+music: effective gain 0.02 -> commands understood,
# 0.03 -> unreliable, >=0.04 -> not understood (and 0.35, the old default, -> nothing heard). So the user-facing
# volume % is mapped quadratically: gain = MUSIC_MAX_GAIN * (pct/100)^2 * loudness_norm. Default 15% -> ~0.008.
MUSIC_MAX_GAIN = float(os.getenv("MUSIC_MAX_GAIN", "0.35"))
MUSIC_DEFAULT_VOLUME_PCT = int(os.getenv("MUSIC_DEFAULT_VOLUME_PCT", "35"))  # was 15; +2 voice steps of 10%
# Music on the HDMI monitor speaker (isolated from the USB mic/speaker housing). MUSIC_TO_HDMI=0 mixes music into the USB
# speaker again. HDMI gain ceiling is higher than MUSIC_MAX_GAIN because voice commands no longer have to be heard over it.
MUSIC_TO_HDMI = os.getenv("MUSIC_TO_HDMI", "1") != "0"
MUSIC_HDMI_MAX_GAIN = float(os.getenv("MUSIC_HDMI_MAX_GAIN", "1.0"))
MUSIC_REF_RMS = float(os.getenv("MUSIC_REF_RMS", "0.149"))   # loudness every track is normalised to (track 1 was 0.149)

# Debug dump of the exact audio sent to Gemini -> tests/recordings/live_sent_turn_N.wav.
# OFF by default (the cropping bug it was built for is fixed). Enable with ME3_SAVE_SENT_AUDIO=1.
SAVE_SENT_AUDIO = os.getenv("ME3_SAVE_SENT_AUDIO", "0").lower() in ("1", "true", "yes")

# Backchannel tolerance: short sounds over the agent ("yeah", "uh-huh", a cough) must NOT interrupt it.
# Voice barge-in needs this much sustained speech. ME3_BACKCHANNEL=0 restores the old ~64 ms trigger.
BACKCHANNEL_ENABLED = os.getenv("ME3_BACKCHANNEL", "1").lower() in ("1", "true", "yes")
BARGE_IN_MIN_MS = int(os.getenv("BARGE_IN_MIN_MS", "450")) if BACKCHANNEL_ENABLED else 64

# -----------------------------------------------------------------------------
# HARDWARE PIN MAP (All 3.3V Logic on RPi 5)
# -----------------------------------------------------------------------------
MOCK_HARDWARE = os.getenv("MOCK_HARDWARE", "0").lower() in ("1", "true", "yes")

# I2C Bus 1
I2C_BUS = 1
ADDR_LCD = 0x27               # 20x4 PCF8574 I2C LCD

# TM1637 4-Digit 7-Segment Display
PIN_TM1637_CLK = 23           # GPIO 23 (Pin 16)
PIN_TM1637_DIO = 24           # GPIO 24 (Pin 18)

# Single RGB Status LED (Common Cathode to Ground)
PIN_RGB_RED = 17              # GPIO 17 (Pin 11)
PIN_RGB_GREEN = 27            # GPIO 27 (Pin 13)
PIN_RGB_BLUE = 22             # GPIO 22 (Pin 15)

# Grove WS2813 Mini NeoPixel Strip via SPI0 MOSI
SPI_LED_STRIP_DEV = "/dev/spidev0.0"  # GPIO 10 MOSI (Pin 19)
NUM_LEDS_STRIP = 10

# Active Buzzer & Push Button
PIN_BUZZER = 6                # GPIO 6 (Pin 31)
PIN_BUTTON = 5                # GPIO 5 (Pin 29, Internal Pull-Up, Active LOW)

# -----------------------------------------------------------------------------
# OPEN-METEO WEATHER CONFIGURATION (Free, Keyless REST API)
# -----------------------------------------------------------------------------
WEATHER_LATITUDE = float(os.getenv("WEATHER_LATITUDE", "14.5995"))    # Manila default
WEATHER_LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "120.9842"))
WEATHER_CITY_NAME = os.getenv("WEATHER_CITY_NAME", "Manila")

# -----------------------------------------------------------------------------
# TELEPHONY & ADB CONSTANTS
# -----------------------------------------------------------------------------
CALL_NUMBER = os.getenv("CALL_NUMBER", "+63XXXXXXXXXX")  # real number lives in .env
MESSAGE_NUMBER = os.getenv("MESSAGE_NUMBER", "+63XXXXXXXXXX")  # real number lives in .env
MESSAGE_TEXT = os.getenv("MESSAGE_TEXT", "Hello from my Raspberry Pi 5 AI Assistant!")
CALL_CARRIER = os.getenv("CALL_CARRIER", "Globe")

# -----------------------------------------------------------------------------
# FILE STORAGE & LOGGING PATHS
# -----------------------------------------------------------------------------
STATE_FILE_PATH = BASE_DIR / "assistant_state.json"
BENCHMARK_LOG_PATH = BASE_DIR / "benchmark_turns.csv"
SERVED_PAGES_DIR = BASE_DIR / "served_pages"
MUSIC_DIR = BASE_DIR / "media"
WORKSPACE_DIR = BASE_DIR / "workspace"
GUI_SOCKET_PATH = Path(os.getenv("GUI_SOCKET_PATH", "/run/user/1000/berry_gui.sock"))
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
SERVED_PAGES_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
