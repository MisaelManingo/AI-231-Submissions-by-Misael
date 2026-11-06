# AI 231 - ME3: Full-Duplex Voice-Enabled AI Agent (Raspberry Pi 5)

An always-listening, full-duplex, interruptible voice assistant running on a Raspberry Pi 5 (8 GB) using **Pipecat**, Google's native **Gemini Live Multimodal Speech-to-Speech API** (`models/gemini-3.8-live`), on-device **Silero VAD**, and local **Model Context Protocol (MCP)** tools.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       G-MARK Micro Go USB Audio       |
                      |  48 kHz Hardware Input / Output (ALSA)|
                      +-------------------+-------------------+
                                          |
                        3:1 Decimation    |    Upsample 1:3 + Stereo
                         (48k -> 16k)     |        (16k -> 48k)
                                          v
+---------------------------------------------------------------------------------+
|                                 Pipecat Harness                                 |
|                                                                                 |
|   +--------------------------+           +----------------------------------+   |
|   |   GMarkAudioTransport    |           |    LocalVADBargeInProcessor      |   |
|   |  - Double-buffered I/O   | --------> |   - Silero VAD ONNX (16 kHz)     |   |
|   |  - Instant queue flush   |           |   - Local turn boundary detect   |   |
|   |  - Echo suppression gate |           |   - Fast barge-in cancellation   |   |
|   +--------------------------+           +-----------------+----------------+   |
|                 ^                                          |                    |
|                 |                                          v                    |
|   +-------------+------------+           +-----------------+----------------+   |
|   |  LLMAssistantAggregator  |           |        LLMUserAggregator         |   |
|   +-------------+------------+           +-----------------+----------------+   |
|                 ^                                          |                    |
|                 |         OutputAudioRawFrame              v                    |
|   +-------------+------------------------------------------+----------------+   |
|   |                          GeminiLiveLLMService                           |   |
|   |                  Model: models/gemini-3.8-live (bidi)                   |   |
|   |                          Voice: Aoede / Puck                            |   |
|   +-------------------------------------+-----------------------------------+   |
+-----------------------------------------|---------------------------------------+
                                          |
                      Function Calling    |    Tool Execution Results
                                          v
+---------------------------------------------------------------------------------+
|                       Local FastMCP Server & Bridge                             |
|                                                                                 |
|  - Media Control (play/pause/stop/volume)                                       |
|  - Live Weather (Open-Meteo REST API, no DHT22) & RPi Internet Time/Date        |
|  - Smart Lights (WS2813 10-NeoPixel strip on/off, brightness, colors)           |
|  - Timers & Alarms (TM1637 countdown, buzzer alert, atomic JSON persistence)    |
|  - Virtual Thermostat (software setpoint/mode + outdoor weather comparison)     |
|  - 24-Hour Expiring Reminders (atomic JSON persistence)                         |
|  - Telephony & Messaging (Android ADB voice call & SMS)                         |
|  - Agentic Website Builder (generates HTML & serves preview at http://localhost)|
|  - General Agent Tasks (safe bash shell commands & file management)             |
+---------------------------------------------------------------------------------+
```

---

## Hardware Pin Map (All 3.3V Logic on Raspberry Pi 5)

| Peripheral | Interface / Pin | Details |
|---|---|---|
| **Audio In / Out** | USB (ALSA Card 2) | G-MARK Micro Go USB microphone & speaker (48 kHz native) |
| **Status LCD** | I2C Bus 1 (SDA: GPIO 2, SCL: GPIO 3) | 20x4 PCF8574 I2C LCD at address `0x27` (Headless Status) |
| **4-Digit Display** | GPIO 23 (CLK), GPIO 24 (DIO) | TM1637 Display (System Clock, Timer countdown, Temp popup) |
| **Status RGB LED** | Red: GPIO 17, Green: GPIO 27, Blue: GPIO 22 | Common Cathode to GND (Yellow=Listen, Blue=Think, Green=Speak) |
| **NeoPixel Strip** | SPI0 MOSI (GPIO 10, `/dev/spidev0.0`) | 10x WS2813 Mini NeoPixels for smart room lighting |
| **Active Buzzer** | GPIO 6 | Timer & Alarm audible alert patterns |
| **Push Button** | GPIO 5 | Internal pull-up, active low; silences buzzer alerts |
| **Android Phone** | USB ADB | UIAutomator2 telephony for voice calls & SMS |

---

## Directory Structure

```
ME 3 - Full-duplex Agent/
├── .env                       # API keys (GOOGLE_API_KEY, GEMINI_LIVE_MODEL, etc.)
├── .gitignore                 # Protected secrets & temporary files
├── requirements-me3.txt       # Dependencies added to existing venv
├── config.py                  # Single unified configuration
│
├── hardware/                  # Modular hardware drivers with mock backend
│   ├── __init__.py
│   ├── displays.py            # 20x4 I2C LCD & TM1637 4-digit display controller
│   ├── leds.py                # Status RGB LED & Grove WS2813 NeoPixel strip
│   ├── alerts.py              # Active Buzzer & Push Button listener
│   ├── state_manager.py       # Atomic JSON persistence (assistant_state.json)
│   ├── telephony.py           # Android ADB calling and SMS
│   └── manager.py             # HardwareManager facade coordinating peripherals
│
├── mcp_server/                # Model Context Protocol action server & bridge
│   ├── __init__.py
│   ├── server.py              # FastMCP server declaring all 10 use-cases + general tools
│   └── client_bridge.py       # Bridges MCP tools to Gemini Live function calling
│
├── agent/                     # Pipecat full-duplex pipeline
│   ├── __init__.py
│   ├── audio_transport.py     # 48 kHz <-> 16 kHz sounddevice transport with barge-in flush
│   ├── vad.py                 # Silero ONNX VAD & echo-suppressed barge-in controller
│   └── pipeline.py            # Main pipeline coordinator
│
├── benchmark/                 # Latency benchmark and instrumentation
│   ├── __init__.py
│   ├── logger.py              # High-precision turn timestamp logger (CSV writer)
│   └── report.py              # Benchmark reporter (median, p95, tool overhead, barge-in)
│
├── tests/                     # Incremental unit & hardware tests
│   ├── test_hardware_subsystems.py # Tests GPIO, I2C, SPI, and mock modes
│   ├── test_audio_loopback.py      # Tests G-MARK USB mic & speaker
│   ├── test_mcp_standalone.py      # Tests all MCP tools standalone
│   ├── test_gemini_connection.py   # Tests Gemini Live WebSocket connection
│   └── test_agent_lifecycle.py     # Tests agent startup, loop, and graceful stop
│
├── run_agent.py               # Main application entrypoint
└── README.md                  # Complete documentation
```

---

## Setup & Installation

The assistant runs in your existing Python virtual environment at `~/AI_231_venv/`:

```bash
# 1. Activate your virtual environment
source ~/AI_231_venv/bin/activate

# 2. Navigate to project directory
cd "/home/misael/MEng AI/AI 231/ME 3 - Full-duplex Agent"

# 3. Install required packages (already completed)
pip install -r requirements-me3.txt
```

### Environment Configuration (`.env`)
Ensure `.env` in `"ME 3 - Full-duplex Agent"` contains your Google API key and chosen Live model:
```env
GOOGLE_API_KEY=your_actual_key_here
GEMINI_LIVE_MODEL=models/gemini-3.8-live
GEMINI_VOICE_ID=Aoede
```

---

## Running the Assistant

### Live Physical Mode (on Raspberry Pi 5)
```bash
source ~/AI_231_venv/bin/activate
python run_agent.py
```
- **LCD Status Display** will show `[LISTENING]` and live RMS meter.
- **TM1637 Display** will show the current internet-synced system clock (`HH:MM`).
- **Status RGB LED** turns Yellow.
- Speak naturally into the G-MARK Micro Go microphone.
- The assistant responds in real-time speech.
- **Barge-in**: Speak while the agent is talking to immediately silence playback and take over the turn.
- Press **Ctrl+C** to exit cleanly. The benchmark report will automatically print upon shutdown.

### Simulation / Mock Mode (Laptop or No Hardware)
```bash
python run_agent.py --mock
```

---

## Testing & Verification Suite

You can test individual subsystems independently:

```bash
# 1. Verify all physical hardware (LCD, TM1637, NeoPixels, Status LED, Buzzer, Button)
python tests/test_hardware_subsystems.py

# 2. Verify G-MARK Micro Go 48 kHz capture & playback
python tests/test_audio_loopback.py

# 3. Verify all MCP tools standalone (weather, time, lights, timers, alarms, thermostat, web builder)
python tests/test_mcp_standalone.py

# 4. Verify Gemini Live WebSocket handshake
python tests/test_gemini_connection.py

# 5. Verify full agent startup and shutdown lifecycle
python tests/test_agent_lifecycle.py

# 6. Interactive Conference Mic/Speaker Voice Calibration & Gemini Live Test
python tests/calibrate_voice.py
```

---

## Latency Benchmark

Every turn records high-precision timestamps (`time.perf_counter`) into `benchmark_turns.csv`:
- `user_speech_end_ts`: Local Silero VAD speech cutoff
- `model_first_audio_ts`: First raw audio packet received from Gemini Live
- `speaker_first_audio_ts`: First audio frame rendered on G-MARK speaker
- `tool_start_ts` & `tool_finish_ts`: Tool execution overhead
- `barge_in_stop_ms`: Time from user barge-in onset to speaker silence

### Generate Benchmark Report
```bash
python benchmark/report.py
```

Example Output:
```
======================================================================
AI 231 ME3: FULL-DUPLEX VOICE AGENT LATENCY BENCHMARK REPORT
======================================================================
1. END-OF-SPEECH TO FIRST-AUDIO-PLAYED LATENCY:
   - Median (p50):     428.5 ms
   - 95th %ile (p95):  612.0 ms
   - Min / Max:        385.0 ms / 740.0 ms

2. NETWORK & SERVER TTFT (EOS to First Cloud Audio Chunk):
   - Median (p50):     382.1 ms
   - 95th %ile (p95):  545.0 ms

3. LOCAL SPEAKER BUFFERING / RESAMPLING LATENCY:
   - Median (p50):     42.4 ms
   - 95th %ile (p95):  68.0 ms

4. BARGE-IN REACTION TIME (User Speech Onset to Playback Silent):
   - Median (p50):     18.2 ms
   - 95th %ile (p95):  32.0 ms

5. TOOL-CALL OVERHEAD BREAKDOWN:
   Tool Name                    | Count  | Median     | p95        | Mean      
   --------------------------------------------------------------------
   control_lights               | 4      | 3.2ms      | 5.1ms      | 3.6ms     
   get_weather                  | 3      | 142.0ms    | 185.0ms    | 151.2ms   
   control_thermostat           | 2      | 138.5ms    | 160.0ms    | 145.0ms   
======================================================================
```

---

## Supported Voice Actions & MCP Tools

1. **Media Control**:
   - *"Play some music"*, *"Pause playback"*, *"Next song"*, *"Turn volume up by 20 percent"*.
2. **Weather & Time/Date**:
   - *"What's the weather like?"* (Queries Open-Meteo REST API for live temperature, humidity, wind, and conditions).
   - *"What time is it?"* (Reports internet-synced system time).
3. **Lights Control**:
   - *"Turn on the lights"*, *"Dim the lights to 40 percent"*, *"Change light color to blue / red / green"*.
   - Any color works (*"make it sunset orange"*, *"hot pink"*, *"lavender"*): the model works out a hex code and the tool also accepts CSS names, `rgb()`, `hsl()` and `r,g,b`; the 8 basic names keep their LED-tuned values.
4. **Timers**:
   - *"Set a timer for 30 seconds"* (Starts countdown on TM1637 and sounds buzzer when finished).
   - *"Cancel the timer"* (Cancels countdown and silences buzzer).
5. **Alarms**:
   - *"Set an alarm for 7:30 AM"* (Saved persistently in `assistant_state.json`; buzzer sounds at alarm time).
   - *"What time is my alarm?"*, *"Cancel the alarm"*.
6. **Virtual Thermostat**:
   - *"Set the thermostat to 22 degrees"* (Updates software setpoint & mode, shows popup on TM1637 and LCD, and compares against current outdoor weather).
7. **Reminders**:
   - *"Remind me to submit my AI 231 report"* (Saved with 24-hour auto-expiration in `assistant_state.json`).
   - *"What are my reminders?"*, *"Clear reminder for submitting report"*.
8. **Telephony & Messaging**:
   - *"Call my phone"*, *"Send an SMS message saying hello"*.
9. **Agentic Website Creator**:
   - *"Create a website for my AI project"* (Generates a styled HTML file in `served_pages/` and serves a local preview at `http://localhost:8080`).
10. **General Agent Tasks**:
    - *"Run shell command uptime"*, *"Check disk space"*, *"Read file test_note.txt"*.
11. **Long-Running Task Delegation (Google Antigravity Agent)**:
    - *"Delegate this complex task: analyze my dataset and generate a summary report"*.
    - Dispatched non-blockingly in the background to `google.antigravity.Agent` powered by **Gemini 3.8 Flash** (`ThinkingLevel.LOW` / `ThinkingLevel.MEDIUM`).
    - The full-duplex voice pipeline remains responsive; you can query progress anytime with *"What is the status of my background task?"*.


---

## GUI, Live Captions, Bilingual Mode & HDMI Music (Nov 2026 update)

Launch everything with `./start_all.sh` (or the **Berry Good Listener** desktop shortcut, which runs `launch_desktop.sh` and
logs to `~/.local/state/berry/last_run.log`). `./start_all.sh --gui-only` shows only the GUI.

| Feature | What it does | Switch |
|---|---|---|
| State colours | Fullscreen background per state: idle white, listening raspberry red, thinking violet, speaking teal; the status LED matches | `ME3_LED_MATCH_GUI=0` restores the old LED colours |
| AI transcript | Bottom of screen: what Berry is saying, streamed per fragment; dimmed with "…" when interrupted; ignored turns never show | `ME3_GUI_TRANSCRIPT=0` hides both transcripts |
| User transcript | Top of screen: rolling last 3 lines, older lines fade; speech not addressed to Berry is shown dimmed/italic | same flag |
| Live captions | Interim (speculative) text while you speak, from a second `gemini-3.5-transcribe-live` session fed the same mic audio; the final text is still the main session's | `ME3_LIVE_CAPTIONS=0`, `ME3_CAPTION_DEBUG=1` |
| Idle keepalive | Sends 100 ms of silence after 45 s idle so the Live session is not closed for inactivity | `ME3_KEEPALIVE_S=0` disables |
| English + Filipino | Understands and replies in English, Filipino or Taglish, matching the user's language | `ME3_LANGS=en` restores English-only |
| Emotional awareness | Gemini 3.8 Live has affective dialog on by default; the prompt asks Berry to use it | `ME3_EMPATHY=0` |
| HDMI music | Music is played on the HDMI monitor speaker so the USB mic/speaker housing does not hear it; falls back to USB if HDMI is unavailable | `MUSIC_TO_HDMI=0`, `MUSIC_HDMI_MAX_GAIN` |

Design rule for all of the above: nothing slow runs in PortAudio callbacks or on the asyncio loop (transcripts are queued and
sent by the existing 25 Hz publisher task; the caption and HDMI streams use their own tasks/threads), and every addition is
behind an environment flag.

> Public-repo notes: `.env` (API key), the music files in `media/`, and `tests/recordings/` are not included. Copy
> `.env.example` to `.env`, add your own `GEMINI_API_KEY`, and put your own tracks in `media/`. The phone numbers in
> `config.py` are placeholders; set `CALL_NUMBER` / `MESSAGE_NUMBER` in `.env`.

### Optional voice tools (`mcp_server/extras.py`)

| Tool | What it does | Switch / settings |
|---|---|---|
| `web_search` | Web or news search: a separate Gemini call with Google Search grounding, returns <= 400 characters to read aloud (about 7-10 s) | `ME3_WEB_SEARCH=0` disables; `SEARCH_MODEL`, `SEARCH_MAX_CHARS`, `SEARCH_TIMEOUT_S` (min 10) |
| `generate_image` | Draws a picture with Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`) in a background thread (about 4 s), saves it to `workspace/images/` and shows it as a fullscreen GUI overlay (sent to the GUI as a file path over the existing socket). Berry asks for a voice yes first (uses API quota) | `ME3_IMAGE_GEN=0` disables; `IMAGE_MODEL`, `IMAGE_SHOW_S` (15), `IMAGE_TIMEOUT_S` |
| `generate_sound` | Generates a ~30 s music/sound clip with Lyria 3 Clip (`lyria-3-clip-preview`, 44.1 kHz stereo MP3, about 12 s) in a background thread and saves it to `workspace/sounds/`. Not played automatically (playback design pending). Berry asks for a voice yes first | `ME3_SOUND_GEN=0` disables; `SOUND_MODEL`, `SOUND_TIMEOUT_S` |
| `spotify_search` | Finds songs on Spotify (Web API client credentials, top 3 matches). Search only: previews are gone for new apps and playback is a separate step | `ME3_SPOTIFY=0` disables; needs `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` in `.env`; `SPOTIFY_MARKET` (PH) |
| *(sound playback)* | A generated sound plays once automatically on the HDMI speaker via `MusicPlayer.play_oneshot`, at most the default music volume x 0.8, ducked under Berry's voice, then the previous music state resumes. Never on the USB speaker (saved only if HDMI is down) | `SOUND_AUTOPLAY=0`, `SOUND_PLAY_GAIN` |
| `spotify_play` / `spotify_control` | Plays a Spotify song on the HDMI speaker (Premium): librespot (official Raspotify build unpacked in `~/.local/share/berry-librespot/`, nothing system-wide) acts as a Connect device named `Berry`; its `pipe` output is read by the HDMI writer thread (`hardware/spotify_source.py`), resampled 44.1 -> 48 kHz and mixed like local music (same volume, ducked under Berry's voice). Berry controls it through the Web API (`mcp_server/spotify_player.py`). Local music and Spotify never play together | `ME3_SPOTIFY_PLAY=0` disables; `SPOTIFY_DEVICE_NAME`, `LIBRESPOT_BIN`, `LIBRESPOT_CACHE_DIR`, `SPOTIFY_TOKEN_FILE`. One-time setup: pair librespot at spotify.com/pair, then `python -m mcp_server.spotify_player auth`. The login cache and token stay in `~/.local/share/berry-librespot/` (never in the repo) |
| *(Spotify backend)* | Preferred playback backend is **Spotify Soloist** (Spotify's official headless client; arm64 build from developer.spotify.com in `~/.local/share/berry-soloist/`, never committed or redistributed, builds expire 90 days after their build date -> exit code 10). Soloist only outputs to PipeWire, so it plays into a virtual sink `berry_spotify` (created at runtime with `pw-cli`, no config file) and `pw-record` captures it for the HDMI writer thread. librespot stays as an unused fallback (Spotify refuses it audio keys). One-time: `SOLOIST_API_KEY` in `.env` (personal, Premium), start Soloist once, pick the device `Berry` in the Spotify app, then `touch ~/.local/share/berry-soloist/paired` | `ME3_SPOTIFY_PLAY=0` disables; logs in `~/.local/state/berry/soloist.log` |
