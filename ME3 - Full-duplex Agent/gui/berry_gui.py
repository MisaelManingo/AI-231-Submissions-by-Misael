#!/usr/bin/env python3
"""
Berry Good Listener - Minimalist Fullscreen GUI
Raspberry Pi 5 voice assistant visual interface.

Design Principles:
- Pure minimalism: plain white background (#FFFFFF).
- Official Raspberry Pi logo centered (assets/logo.svg).
- Option B (Equatorial Horizon): Animated audio waveform running through the horizontal center.
- Palette: Official Raspberry Pi Brand Red (#C51A4A), Warm Berry Rose (#E87092), Soft Blush (#F9E8ED).
- State-driven animation without words:
    * Idle: Flat subtle line.
    * Listening (User Speaking): Brand red (#C51A4A) dynamic waveform driven by mic RMS.
    * Thinking: Soft rose (#E87092) gently pulsing breathing flatline (~0.8 Hz).
    * Speaking (Agent Voice): Warm rose (#E87092) fluid waveform driven by speaker level.
- Decoupled process: listens for datagrams on /run/user/1000/berry_gui.sock.
"""

import os
import sys

# Prevent SDL from touching or acquiring ALSA / audio hardware
os.environ["SDL_AUDIODRIVER"] = "dummy"

import math
import time
import json
import signal
import socket
import argparse
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Socket path (defaults to XDG_RUNTIME_DIR / berry_gui.sock)
GUI_SOCKET_PATH = Path(os.getenv("GUI_SOCKET_PATH", f"/run/user/{os.getuid()}/berry_gui.sock"))

# -----------------------------------------------------------------------------
# PALETTE DEFINITIONS (Official Raspberry Pi Brand Colors)
# -----------------------------------------------------------------------------
COLOR_BG = (255, 255, 255)            # Pure White default
COLOR_BRAND_RED = (197, 26, 74)       # Raspberry Pi Red (#C51A4A)
COLOR_WARM_ROSE = (232, 112, 146)     # Warm Berry Rose (#E87092)
COLOR_SOFT_BLUSH = (249, 232, 237)    # Soft Blush Tint (#F9E8ED)
COLOR_IDLE_LINE = (225, 225, 230)     # Muted tranquil reference line
COLOR_LOGO_DARK = (17, 25, 28)        # Embedded Logo Silhouette (#11191C)

# Background State Tints (Smoothly blended into white)
BG_IDLE = (255, 255, 255)             # Pure White
# Scheme A (saturated, readable across a room). Previous pale tints, for easy revert:
#   LISTENING (252,232,237)  THINKING (244,236,250)  SPEAKING (252,238,244)
BG_LISTENING = (197, 26, 74)          # #C51A4A raspberry red
BG_THINKING = (91, 63, 168)           # #5B3FA8 deep violet
BG_THINKING_LO = (72, 48, 140)        # darker violet for the breathing pulse
BG_SPEAKING = (30, 136, 168)          # #1E88A8 teal
COLOR_WAVE_ON_COLOR = (255, 255, 255) # waveform/text colour on saturated states
BG_LERP = 0.25                        # was 0.15; ~150 ms settle at 30 FPS


# Transcript overlay flag (same env var the agent reads). ME3_GUI_TRANSCRIPT=0 hides the overlay.
TRANSCRIPT_ENABLED = os.getenv("ME3_GUI_TRANSCRIPT", "1") != "0"
TR_FONT_PX = 26
TR_MAX_LINES = 3
TR_MARGIN = 40
AI_HOLD_S = 4.0       # keep a finished reply this long, then fade
AI_FADE_S = 1.0
AI_STALE_S = 8.0      # streaming with no new fragment this long => treat as finished


class AiTranscript:
    """Bottom-of-screen text of what Berry is saying. Pure state + wrapping; drawing is done by BerryGui."""

    def __init__(self):
        self.text = ""
        self.status = "idle"     # "streaming" | "done" | "interrupted" | "idle"
        self.ts = 0.0            # time of last change

    def apply(self, op: str, text: str, now: float):
        if op == "append":
            if self.status != "streaming":   # first fragment of a new reply replaces the old one
                self.text = ""
            self.status = "streaming"
            self.text += text
            self.ts = now
        elif op == "end":
            if self.status == "streaming":
                self.status = "done"
                self.ts = now
        elif op == "interrupt":
            if self.status == "streaming" and self.text:
                self.text = self.text.rstrip() + "\u2026"
                self.status = "interrupted"
                self.ts = now
        elif op == "ignore":                  # attention gate: this reply must not be shown
            self.text, self.status = "", "idle"

    def alpha(self, now: float) -> float:
        """0..1 opacity; also retires stale/finished text."""
        if not self.text:
            return 0.0
        age = now - self.ts
        if self.status == "streaming":
            if age > AI_STALE_S:
                self.status, self.ts = "done", now
            return 1.0
        base = 0.5 if self.status == "interrupted" else 1.0   # interrupted replies are dimmed
        hold = 1.5 if self.status == "interrupted" else AI_HOLD_S
        if age <= hold:
            return base
        fade = 1.0 - (age - hold) / AI_FADE_S
        if fade <= 0.0:
            self.text, self.status = "", "idle"
            return 0.0
        return base * fade


USER_HOLD_S = 8.0     # a user line stays this long after it arrives, then fades
USER_FADE_S = 1.0
USER_RANK_ALPHA = (1.0, 0.75, 0.55)  # newest line -> older lines
USER_PREVIEW_STALE_S = 4.0           # drop a live preview not updated/finalized for this long
USER_IGNORED_ALPHA = 0.7             # ignored (not addressed to Berry): dimmed + italic


class UserTranscript:
    """Top-of-screen rolling transcript of what the user said. Entries are sentence chunks."""

    def __init__(self):
        self.entries = []        # dicts: text, ts, ignored
        self.turn_start = 0      # index of the first entry of the current (not yet answered) user turn
        self.preview = ""        # speculative live text (interim); replaced by each update, cleared by the final text
        self.preview_ts = 0.0

    def apply(self, op: str, text: str, now: float):
        if op == "interim":
            self.preview, self.preview_ts = text.strip(), now
        elif op == "add" and text.strip():
            self.preview = ""                         # final text supersedes the speculative preview
            self.entries.append({"text": text.strip(), "ts": now, "ignored": False})
            if len(self.entries) > 12:               # bounded; only the last few lines are ever shown
                drop = len(self.entries) - 12
                del self.entries[:drop]
                self.turn_start = max(0, self.turn_start - drop)
        elif op == "ignore":                          # attention gate: dim this turn's lines
            self.preview = ""
            for e in self.entries[self.turn_start:]:
                e["ignored"] = True
            self.turn_start = len(self.entries)

    def new_turn(self):
        """Berry started/ended a reply, so later user chunks belong to a new turn."""
        self.preview = ""
        self.turn_start = len(self.entries)

    def visible(self, now: float):
        """[(text, ignored, age_alpha)] oldest first; retires fully faded entries."""
        out = []
        keep = []
        for e in self.entries:
            age = now - e["ts"]
            a = 1.0 if age <= USER_HOLD_S else 1.0 - (age - USER_HOLD_S) / USER_FADE_S
            if a > 0.0:
                keep.append(e)
                out.append((e["text"], e["ignored"], a))
        if self.preview and now - self.preview_ts <= USER_PREVIEW_STALE_S:
            out.append((self.preview, False, 1.0))     # live preview is the newest line
        elif self.preview:
            self.preview = ""                          # stale: final text never came
        if len(keep) != len(self.entries):
            self.turn_start = max(0, self.turn_start - (len(self.entries) - len(keep)))
            self.entries = keep
        return out


def wrap_text(font, text: str, max_w: int):
    """Greedy word wrap using the font's metrics. Very long words are hard-split."""
    lines, cur = [], ""
    for word in text.split():
        trial = word if not cur else cur + " " + word
        if font.size(trial)[0] <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
        while font.size(cur)[0] > max_w and len(cur) > 1:
            k = len(cur)
            while k > 1 and font.size(cur[:k])[0] > max_w:
                k -= 1
            lines.append(cur[:k])
            cur = cur[k:]
    if cur:
        lines.append(cur)
    return lines


class BerryGui:
    """Minimalist fullscreen GUI for Berry Good Listener."""

    def __init__(self, fullscreen: bool = True, width: int = 1024, height: int = 600, test_mode: bool = False):
        self.fullscreen = fullscreen
        self.target_width = width
        self.target_height = height
        self.test_mode = test_mode
        self.running = True

        # State tracking
        self.state = "idle"         # "idle", "listening", "thinking", "speaking"
        self.level = 0.0            # Target level (0.0 to 1.0)
        self.smoothed_level = 0.0   # Interpolated level for fluid motion
        self.last_msg_ts = time.time()
        self.phase = 0.0
        self.current_bg = [255.0, 255.0, 255.0]
        self.ai_tr = AiTranscript()
        self.user_tr = UserTranscript()
        self._user_cache = {}
        self._tr_cache = None       # (key, [surfaces]) so text is re-rendered only when it changes

        # Socket setup
        self.sock = None
        if not self.test_mode:
            self._setup_socket()

        # Pygame setup (display only, no audio)
        import pygame
        self.pygame = pygame
        self.pygame.display.init()
        self.pygame.display.set_caption("Berry Good Listener")

        # Configure display mode
        flags = pygame.NOFRAME
        if self.fullscreen:
            # Under Wayland / labwc, set mode to available native resolution
            info = pygame.display.Info()
            self.width = info.current_w if info.current_w > 0 else self.target_width
            self.height = info.current_h if info.current_h > 0 else self.target_height
            flags |= pygame.FULLSCREEN
        else:
            self.width = self.target_width
            self.height = self.target_height

        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        self.pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.tr_font, self.tr_font_italic = self._load_tr_fonts(pygame)

        # Load and prepare official logo
        self._load_logo()

        # Register signals for graceful termination
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _setup_socket(self):
        """Prepares the AF_UNIX datagram socket."""
        try:
            if GUI_SOCKET_PATH.exists():
                GUI_SOCKET_PATH.unlink()
            GUI_SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)

            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.sock.bind(str(GUI_SOCKET_PATH))
            self.sock.setblocking(False)
            print(f"[GUI]: Listening on datagram socket: {GUI_SOCKET_PATH}", flush=True)
        except Exception as e:
            print(f"[GUI]: Socket initialization error: {e}", flush=True)
            self.sock = None

    def _load_logo(self):
        """Loads assets/logo.svg and scales appropriately."""
        logo_path = BASE_DIR / "assets" / "logo.svg"
        if not logo_path.exists():
            print(f"\n❌ FATAL ERROR: Logo file missing at: {logo_path}")
            print("Please provide the official logo file before running the GUI.")
            sys.exit(1)

        try:
            # First attempt to extract high-res embedded PNG from SVG to render complete logo graphics
            raw_surf = None
            content = logo_path.read_text(encoding="utf-8", errors="ignore")
            import re
            import io
            import base64
            m = re.search(r'base64,([^"\'\s]+)', content)
            if m:
                from PIL import Image
                img_data = base64.b64decode(m.group(1))
                pil_img = Image.open(io.BytesIO(img_data)).convert("RGBA")
                raw_surf = self.pygame.image.fromstring(pil_img.tobytes(), pil_img.size, "RGBA")

            if raw_surf is None:
                raw_surf = self.pygame.image.load(str(logo_path)).convert_alpha()

            orig_w, orig_h = raw_surf.get_size()

            # Scale logo proportionally: height around 45% of screen height
            target_h = int(self.height * 0.45)
            target_w = int(orig_w * (target_h / orig_h))
            self.logo = self.pygame.transform.smoothscale(raw_surf, (target_w, target_h))
            self.logo_rect = self.logo.get_rect(center=(self.width // 2, self.height // 2))
            print(f"[GUI]: Loaded official logo ({orig_w}x{orig_h} -> {target_w}x{target_h}) at center {self.logo_rect.center}", flush=True)
        except Exception as e:
            print(f"[GUI]: Error loading logo SVG: {e}", flush=True)
            sys.exit(1)

    def _handle_signal(self, signum, frame):
        self.running = False

    def _poll_socket(self):
        """Drains non-blocking socket and captures newest state."""
        if not self.sock:
            return

        latest_data = None
        while True:
            try:
                data, _ = self.sock.recvfrom(2048)  # was 512; transcript datagrams are larger
                # Transcript datagrams ({"type":"tr"}) are applied in order; state datagrams: newest wins (as before).
                if b'"type"' in data and b'"tr"' in data:
                    self._handle_transcript(data)
                else:
                    latest_data = data
            except (BlockingIOError, OSError):
                break

        if latest_data:
            try:
                msg = json.loads(latest_data.decode("utf-8"))
                self.state = msg.get("state", "idle")
                self.level = float(msg.get("level", 0.0))
                self.last_msg_ts = time.time()
            except Exception:
                pass
        else:
            # Fall back to idle if no messages for > 1.2s
            if time.time() - self.last_msg_ts > 1.2:
                self.state = "idle"
                self.level = 0.0

    @staticmethod
    def _load_tr_fonts(pygame):
        """Roboto Mono (user-installed in ~/.local/share/fonts/RobotoMono); falls back to a system font."""
        d = Path.home() / ".local" / "share" / "fonts" / "RobotoMono"
        try:
            return (pygame.font.Font(str(d / "RobotoMono-Regular.ttf"), TR_FONT_PX),
                    pygame.font.Font(str(d / "RobotoMono-Italic.ttf"), TR_FONT_PX))
        except Exception as e:
            print(f"[GUI]: Roboto Mono not found ({e}); using fallback font.", flush=True)
            f = pygame.font.SysFont("dejavusans,freesans,arial", TR_FONT_PX)
            fi = pygame.font.SysFont("dejavusans,freesans,arial", TR_FONT_PX, italic=True)
            return f, fi

    def _handle_transcript(self, data: bytes):
        """Applies one transcript datagram. Never raises."""
        if not TRANSCRIPT_ENABLED:
            return
        try:
            msg = json.loads(data.decode("utf-8"))
            now = time.time()
            if msg.get("who") == "ai":
                self.ai_tr.apply(msg.get("op", ""), msg.get("text", ""), now)
                if msg.get("op") in ("append", "end", "interrupt"):
                    self.user_tr.new_turn()   # the user's turn has been answered
            elif msg.get("who") == "user":
                self.user_tr.apply(msg.get("op", ""), msg.get("text", ""), now)
        except Exception:
            pass

    def _draw_ai_transcript(self, now: float):
        """Bottom band: last TR_MAX_LINES wrapped lines of Berry's current reply, bottom-anchored."""
        alpha = self.ai_tr.alpha(now)
        if alpha <= 0.0:
            return
        # Dark text on the white idle background, white on the saturated states (discrete, so the cache is stable)
        lum = 0.299 * self.current_bg[0] + 0.587 * self.current_bg[1] + 0.114 * self.current_bg[2]
        color = COLOR_LOGO_DARK if lum > 170 else (255, 255, 255)
        key = (self.ai_tr.text, color)
        if self._tr_cache is None or self._tr_cache[0] != key:
            lines = wrap_text(self.tr_font, self.ai_tr.text, self.width - 2 * TR_MARGIN)[-TR_MAX_LINES:]
            self._tr_cache = (key, [self.tr_font.render(l, True, color) for l in lines])
        surfs = self._tr_cache[1]
        lh = self.tr_font.get_linesize()
        y = self.height - TR_MARGIN - lh * len(surfs)
        for surf in surfs:
            surf.set_alpha(int(255 * alpha))
            self.screen.blit(surf, ((self.width - surf.get_width()) // 2, y))
            y += lh

    def _draw_user_transcript(self, now: float):
        """Top band: last TR_MAX_LINES wrapped user lines, newest at the bottom of the block, older ones fainter."""
        vis = self.user_tr.visible(now)
        if not vis:
            return
        lum = 0.299 * self.current_bg[0] + 0.587 * self.current_bg[1] + 0.114 * self.current_bg[2]
        color = COLOR_LOGO_DARK if lum > 170 else (255, 255, 255)
        max_w = self.width - 2 * TR_MARGIN
        lines = []   # (text, italic, entry_alpha)
        for text, ignored, a in vis:
            font = self.tr_font_italic if ignored else self.tr_font
            for l in wrap_text(font, text, max_w):
                lines.append((l, ignored, a))
        lines = lines[-TR_MAX_LINES:]
        lh = self.tr_font.get_linesize()
        y = TR_MARGIN + lh * (TR_MAX_LINES - len(lines))   # bottom-anchored inside the block
        cache = self._user_cache
        if len(cache) > 40:
            cache.clear()
        for i, (l, ignored, a) in enumerate(lines):
            rank = len(lines) - 1 - i                       # 0 = newest
            alpha = USER_RANK_ALPHA[min(rank, len(USER_RANK_ALPHA) - 1)] * a
            if ignored:
                alpha *= USER_IGNORED_ALPHA
            k = (l, ignored, color)
            surf = cache.get(k)
            if surf is None:
                surf = cache[k] = (self.tr_font_italic if ignored else self.tr_font).render(l, True, color)
            surf.set_alpha(int(255 * alpha))
            self.screen.blit(surf, ((self.width - surf.get_width()) // 2, y))
            y += lh

    def _update_test_mode(self, t: float):
        """Cycles through states for visual auditioning without agent."""
        cycle = t % 14.0
        if cycle < 3.0:
            self.state = "idle"
            self.level = 0.0
        elif cycle < 7.0:
            self.state = "listening"
            # Simulated speech modulation
            self.level = 0.25 + 0.45 * max(0.0, math.sin(t * 8.0) * math.sin(t * 3.5))
        elif cycle < 10.0:
            self.state = "thinking"
            self.level = 0.0
        else:
            self.state = "speaking"
            # Simulated assistant speech modulation
            self.level = 0.30 + 0.50 * max(0.0, math.cos(t * 9.0) * math.sin(t * 4.0))

    def _draw_waveform(self, t: float):
        """
        Renders the Option B (Equatorial Horizon) waveform.
        The waveform flows through the center horizon of the screen.
        """
        center_y = self.height // 2
        num_points = 96
        step_x = self.width / (num_points - 1)

        # Smooth level interpolation
        self.smoothed_level += (self.level - self.smoothed_level) * 0.28

        # Determine amplitude and active color based on state
        if self.state == "listening":
            active_color = COLOR_BRAND_RED
            amplitude = 65.0 * self.smoothed_level
            speed = 7.0
            freq1, freq2 = 2.5, 5.0
        elif self.state == "speaking":
            active_color = COLOR_WARM_ROSE
            amplitude = 60.0 * self.smoothed_level
            speed = 6.0
            freq1, freq2 = 3.0, 6.0
        elif self.state == "thinking":
            active_color = COLOR_WARM_ROSE
            # Gentle 0.8 Hz breathing pulse
            pulse = (math.sin(t * 2 * math.pi * 0.8) + 1.0) * 0.5
            amplitude = 4.0 + 6.0 * pulse
            speed = 2.5
            freq1, freq2 = 2.0, 4.0
        else:
            # Idle
            active_color = COLOR_IDLE_LINE
            amplitude = 0.0
            speed = 1.0
            freq1, freq2 = 1.0, 2.0

        # Saturated backgrounds: red/rose waveform would vanish, so draw it white.
        if active_color != COLOR_IDLE_LINE:
            active_color = COLOR_WAVE_ON_COLOR

        self.phase += (1.0 / 30.0) * speed

        # Construct waveform vertices with Hanning-style edge tapering
        points = []
        for i in range(num_points):
            x = i * step_x
            norm_x = i / (num_points - 1)  # 0.0 to 1.0

            # Window envelope: zero at boundaries, 1.0 at center
            window = math.sin(math.pi * norm_x)

            if amplitude > 0.5:
                # Multi-harmonic organic waveform
                w1 = math.sin(norm_x * math.pi * freq1 + self.phase)
                w2 = 0.35 * math.sin(norm_x * math.pi * freq2 - self.phase * 1.3)
                w3 = 0.15 * math.cos(norm_x * math.pi * 8.0 + self.phase * 0.7)
                y = center_y + amplitude * window * (w1 + w2 + w3)
            else:
                y = center_y

            points.append((x, y))

        # 1. Subtle soft blush underglow when active
        if amplitude > 2.0 and active_color != COLOR_IDLE_LINE:
            glow_surface = self.pygame.Surface((self.width, self.height), self.pygame.SRCALPHA)
            glow_color = (*active_color, 45)  # 18% alpha glow
            for dy in (-2, -1, 1, 2):
                shifted_points = [(p[0], p[1] + dy) for p in points]
                self.pygame.draw.lines(glow_surface, glow_color, False, shifted_points, 3)
            self.screen.blit(glow_surface, (0, 0))

        # 2. Main anti-aliased waveform line
        line_width = 3 if (amplitude > 2.0 and active_color != COLOR_IDLE_LINE) else 1
        self.pygame.draw.lines(self.screen, active_color, False, points, line_width)

    def run(self):
        """Main rendering loop locked at 30 FPS."""
        print("[GUI]: Berry Good Listener GUI online.", flush=True)
        t0 = time.time()

        while self.running:
            # Handle window events
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.running = False
                elif event.type == self.pygame.KEYDOWN:
                    if event.key in (self.pygame.K_ESCAPE, self.pygame.K_q):
                        self.running = False

            now = time.time()
            elapsed = now - t0

            if self.test_mode:
                self._update_test_mode(elapsed)
            else:
                self._poll_socket()

            # 1. Fill dynamic state-responsive background color (smooth lerp)
            if self.state == "listening":
                target_bg = BG_LISTENING
            elif self.state == "speaking":
                target_bg = BG_SPEAKING
            elif self.state == "thinking":
                pulse = (math.sin(elapsed * 2 * math.pi * 0.8) + 1.0) * 0.5
                target_bg = tuple(
                    BG_THINKING_LO[c] + (BG_THINKING[c] - BG_THINKING_LO[c]) * pulse
                    for c in range(3)
                )
            else:
                target_bg = BG_IDLE

            for c in range(3):
                self.current_bg[c] += (target_bg[c] - self.current_bg[c]) * BG_LERP

            bg_tuple = (int(self.current_bg[0]), int(self.current_bg[1]), int(self.current_bg[2]))
            self.screen.fill(bg_tuple)

            # 2. Draw Option B waveform across the center horizon
            self._draw_waveform(elapsed)

            # 3. Blit official logo cleanly centered over horizon
            self.screen.blit(self.logo, self.logo_rect)

            # 4. Transcript overlay (drawn last; flag-gated)
            if TRANSCRIPT_ENABLED:
                self._draw_ai_transcript(now)
                self._draw_user_transcript(now)

            # Flip display buffer
            self.pygame.display.flip()
            self.clock.tick(30)

        self.cleanup()

    def cleanup(self):
        """Cleans up display and unlinks socket."""
        print("[GUI]: Closing GUI process...", flush=True)
        try:
            self.pygame.quit()
        except Exception:
            pass
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if not self.test_mode and GUI_SOCKET_PATH.exists():
            try:
                GUI_SOCKET_PATH.unlink()
            except Exception:
                pass
        print("[GUI]: Shutdown complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Berry Good Listener GUI")
    parser.add_argument("--windowed", action="store_true", help="Run windowed (non-fullscreen)")
    parser.add_argument("--width", type=int, default=1024, help="Window width (default: 1024)")
    parser.add_argument("--height", type=int, default=600, help="Window height (default: 600)")
    parser.add_argument("--test-mode", action="store_true", help="Run test animation cycle without agent")
    args = parser.parse_args()

    gui = BerryGui(
        fullscreen=not args.windowed,
        width=args.width,
        height=args.height,
        test_mode=args.test_mode
    )
    gui.run()


if __name__ == "__main__":
    main()
