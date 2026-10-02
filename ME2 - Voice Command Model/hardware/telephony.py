#!/usr/bin/env python3
"""
Android ADB & UIAutomator2 Telephony Driver
Performs phone calls and SMS messaging via connected Android device.
Constants are clearly placed at the top for immediate customization.
"""

import time
import shlex
import threading
import subprocess
from .config import (
    CALL_NUMBER as CFG_CALL_NUMBER,
    MESSAGE_NUMBER as CFG_MSG_NUMBER,
    MESSAGE_TEXT as CFG_MSG_TEXT,
    CALL_CARRIER as CFG_CARRIER,
    MOCK_HARDWARE
)

# =========================================================
# EDITABLE TELEPHONY CONSTANTS (AT THE TOP)
# =========================================================
CALL_NUMBER = CFG_CALL_NUMBER        # Destination phone number for voice call
MESSAGE_NUMBER = CFG_MSG_NUMBER      # Destination phone number for SMS
MESSAGE_TEXT = CFG_MSG_TEXT          # Default text message content
CALL_CARRIER = CFG_CARRIER          # SIM carrier popup match (e.g. "Globe" or "SMART")
# =========================================================

try:
    import uiautomator2 as u2
except (ImportError, Exception):
    u2 = None


class TelephonyController:
    def __init__(self, mock=False):
        self.mock = mock or MOCK_HARDWARE or (u2 is None)
        self.d = None

        if not self.mock:
            try:
                self.d = u2.connect()
                print("[TELEPHONY]: Connected to Android device via UIAutomator2.")
            except Exception as e:
                print(f"[TELEPHONY WARNING]: Could not connect to Android device ({e}). Using mock.")
                self.mock = True

    def _adb(self, *args):
        if self.mock:
            print(f"[MOCK ADB]: adb {' '.join(args)}")
            return ""

        result = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    def make_call_async(self, number=CALL_NUMBER, carrier=CALL_CARRIER):
        """Starts phone call in background thread so the audio loop never blocks."""
        thread = threading.Thread(target=self._call_worker, args=(number, carrier), daemon=True)
        thread.start()

    def _call_worker(self, number, carrier):
        print(f"📞 [TELEPHONY]: Placing call to {number} via {carrier}...")
        if self.mock:
            return

        try:
            self._adb("shell", "am", "start",
                      "-a", "android.intent.action.CALL",
                      "-d", f"tel:{number}")
            if self.d:
                sel = self.d(textContains=f"({carrier})")
                if sel.wait(timeout=5):
                    sel.click()
            print("📞 [TELEPHONY]: Call initiated.")
        except Exception as e:
            print(f"[TELEPHONY CALL ERROR]: {e}")

    def hangup(self):
        """Hangs up current call."""
        if self.mock:
            print("[MOCK TELEPHONY]: Hanging up call.")
            return

        try:
            self._adb("shell", "input", "keyevent", "KEYCODE_ENDCALL")
        except Exception as e:
            print(f"[TELEPHONY HANGUP ERROR]: {e}")

    def send_sms_async(self, number=MESSAGE_NUMBER, text=MESSAGE_TEXT):
        """Sends SMS in background thread so the audio loop never blocks."""
        thread = threading.Thread(target=self._sms_worker, args=(number, text), daemon=True)
        thread.start()

    def _sms_worker(self, number, text):
        print(f"💬 [TELEPHONY]: Sending SMS to {number}: '{text}'...")
        if self.mock:
            return

        try:
            self._adb("shell", "am", "start",
                      "-a", "android.intent.action.SENDTO",
                      "-d", f"sms:{number}",
                      "-p", "com.google.android.apps.messaging",
                      "--es", "sms_body", shlex.quote(text))
            time.sleep(5)
            if self.d:
                self.d(descriptionContains="Send").click()
            print("💬 [TELEPHONY]: SMS sent.")
        except Exception as e:
            print(f"[TELEPHONY SMS ERROR]: {e}")

