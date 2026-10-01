import subprocess
import time
import shlex
import uiautomator2 as u2

d = u2.connect()

def adb(*args):
    result = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()

def call(number, carrier="Globe"):
    """Start a phone call."""
    adb("shell", "am", "start",
        "-a", "android.intent.action.CALL",
        "-d", f"tel:{number}")
#         "--ei", "com.android.phone.extra.slot", str(sim),
#         "--ez", "com.android.phone.force.slot", "true",
#         "--ei", "simSlot", str(sim))
    sel = d(textContains=f"({carrier})")
    if sel.wait(timeout=5):
        sel.click()

def hangup():
    adb("shell", "input", "keyevent", "KEYCODE_ENDCALL")

def send_sms(number, text):
    """Open the SMS composer prefilled, then press Send."""
#     adb("shell", "am", "start", "-a", "android.intent.action.SENDTO",
#         "-d", f"sms:{number}", "--es", "sms_body", text,
#         "--ez", "exit_on_sent", "true")
    adb("shell", "am", "start",
        "-a", "android.intent.action.SENDTO",
        "-d", f"sms:{number}",
        "-p", "com.google.android.apps.messaging",
        "--es", "sms_body", shlex.quote(text))
    time.sleep(5)
#     adb("shell", "input", "keyevent", "KEYCODE_ENTER")
    d(descriptionContains="Send").click()

if __name__ == "__main__":
    TARGET_NUMBER = "+639088152097"
    
#     call(TARGET_NUMBER, "Globe") # SMART or Globe
#     time.sleep(30)
#     hangup()
    
    send_sms(TARGET_NUMBER, "Hello from my Raspberry Pi!")