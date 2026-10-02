#!/usr/bin/env python3
"""
Hardware Manager Facade
Coordinates all peripheral subsystems:
- AudioEngine (Single-stream mixer, music player, chimes, TTS)
- StatusRGBLED (Yellow, Blue, Green, Red indicator)
- GroveLEDStrip (10-NeoPixel strip with brightness & color)
- LCDDisplay (20x4 I2C LCD, double-buffered, flicker-free)
- TM1637Controller (4-Digit Display for RTC clock, temp popup, timer)
- DS3231RTC (Read-only real-time clock)
- DHT22Sensor (Temperature & Humidity with rule evaluation)
- TelephonyController (Android ADB Call & SMS)
- StateManager & AlarmAndTimerService (24h reminders, alarms, buzzer, button)
"""

import os
import time
import threading

from .config import (
    TTS_DIR, CHIME_WAKE_PATH, CHIME_END_PATH
)
from .audio_engine import AudioEngine
from .leds import StatusRGBLED, GroveLEDStrip
from .displays import LCDDisplay, TM1637Controller
from .sensors import DS3231RTC, DHT22Sensor
from .telephony import TelephonyController
from .reminders_alarm import StateManager, AlarmAndTimerService


class HardwareManager:
    def __init__(self, mock=False):
        print("\n" + "=" * 65)
        print("INITIALIZING RASPBERRY PI 5 HARDWARE SUBSYSTEMS")
        print("=" * 65)

        # 1. State Persistence
        self.state_mgr = StateManager()

        # 2. Sensors (RTC & DHT22)
        self.rtc = DS3231RTC(mock=mock)
        self.dht = DHT22Sensor(mock=mock)

        # 3. Audio Engine
        self.audio = AudioEngine(mock=mock)

        # 4. Displays
        self.lcd = LCDDisplay(mock=mock)
        self.tm1637 = TM1637Controller(rtc_reader=self.rtc, mock=mock)

        # 5. LEDs
        self.status_led = StatusRGBLED(mock=mock)
        self.strip = GroveLEDStrip(mock=mock)

        # 6. Telephony
        self.telephony = TelephonyController(mock=mock)

        # 7. Alarm & Timer Service
        self.alarm_timer = AlarmAndTimerService(
            state_mgr=self.state_mgr,
            audio_engine=self.audio,
            display_controller=self.tm1637,
            rtc_sensor=self.rtc,
            mock=mock
        )

        # Initial UI states
        self.status_led.set_yellow()
        self.lcd.set_line(0, 'Say "Hey Raspberry!"')
        self.lcd.set_line(1, "RMS: [..........]   ")
        self.lcd.set_line(2, "Ready for wake word ")
        self.lcd.set_line(3, "RPi 5 Tiny Assistant")
        self.lcd.flush(force=True)

        print("[HARDWARE MANAGER]: All subsystems online.\n")

    # -----------------------------------------------------
    # PIPELINE LIFECYCLE HOOKS
    # -----------------------------------------------------
    def update_idle_display(self, rms, wake_score, wake_thresh):
        """Throttled update of LCD row 1 with RMS meter and live wake confidence."""
        norm_val = min(1.0, max(0.0, rms * 15.0))
        filled = int(norm_val * 10)
        bar = "#" * filled + "." * (10 - filled)
        
        self.lcd.set_line(0, 'Say "Hey Raspberry!"')
        self.lcd.set_line(1, f"RMS:[{bar}] W:{int(wake_score*100):2d}%")
        self.lcd.flush()

    def on_wake_detected(self):
        """
        Executed immediately when wake word is detected:
        1. Duck music so command audio can be captured clearly.
        2. Set Single RGB LED to BLUE.
        3. Update LCD Row 0 to 'Say Command...'.
        4. Play chime_wake.wav and WAIT until it finishes.
        """
        self.audio.duck()
        self.status_led.set_blue()
        self.lcd.set_line(0, "Say Command...      ")
        self.lcd.set_line(2, "Listening 2.0s...   ")
        self.lcd.flush(force=True)

        if os.path.exists(CHIME_WAKE_PATH):
            self.audio.play_sfx(CHIME_WAKE_PATH, wait=True)
        else:
            time.sleep(0.2)

    def on_command_window_ended(self):
        """Plays end chime immediately when command recording completes."""
        if os.path.exists(CHIME_END_PATH):
            self.audio.play_sfx(CHIME_END_PATH, wait=False)

    def on_command_ignored(self, top_label, top_conf, is_background=False, latency_ms=0.0):
        """Handles ignored commands (ambient noise or below threshold)."""
        self.status_led.set_red()
        self.lcd.set_line(0, "Command Ignored     ")
        self.lcd.set_line(2, f"CMD: {top_label[:14]}")
        self.lcd.set_line(3, f"Conf:{top_conf*100:4.1f}% {latency_ms:4.0f}ms")
        self.lcd.flush(force=True)

        if not is_background:
            # Play "Sorry, I didn't catch that"
            sorry_wav = os.path.join(TTS_DIR, "sorry_didnt_catch.wav")
            self.audio.play_sfx(sorry_wav, wait=True)

    def on_command_accepted(self, top_label, top_conf, latency_ms=0.0):
        """Updates LED to GREEN and displays recognized command on LCD."""
        self.status_led.set_green()
        self.lcd.set_line(0, "Executing Command   ")
        self.lcd.set_line(2, f"CMD: {top_label[:14]}")
        self.lcd.set_line(3, f"Conf:{top_conf*100:4.1f}% {latency_ms:4.0f}ms")
        self.lcd.flush(force=True)

    def on_pipeline_cycle_complete(self):
        """Restores music volume, returns LED to YELLOW and resets LCD."""
        self.audio.unduck()
        self.status_led.set_yellow()
        self.lcd.set_line(0, 'Say "Hey Raspberry!"')
        self.lcd.flush()

    # -----------------------------------------------------
    # COMMAND DISPATCH LOGIC
    # -----------------------------------------------------
    def execute_action(self, label, device_state):
        """
        Executes physical action for the label and plays pre-made TTS reply.
        Returns updated device_state dictionary.
        """
        tts_to_play = None
        tts_sequence = None

        # --- LIGHTS ---
        if label == "LIGHT_ON":
            device_state["lights"] = "ON"
            self.strip.turn_on()
            tts_to_play = os.path.join(TTS_DIR, "light_on.wav")

        elif label == "LIGHT_OFF":
            device_state["lights"] = "OFF"
            self.strip.turn_off()
            tts_to_play = os.path.join(TTS_DIR, "light_off.wav")

        elif label == "BRIGHTNESS_20":
            device_state["brightness"] = 20
            self.strip.set_brightness(20)
            tts_to_play = os.path.join(TTS_DIR, "bright_20.wav")

        elif label == "BRIGHTNESS_60":
            device_state["brightness"] = 60
            self.strip.set_brightness(60)
            tts_to_play = os.path.join(TTS_DIR, "bright_60.wav")

        elif label == "BRIGHTNESS_100":
            device_state["brightness"] = 100
            self.strip.set_brightness(100)
            tts_to_play = os.path.join(TTS_DIR, "bright_100.wav")

        elif label == "COLOR_RED":
            device_state["light_color"] = "RED"
            self.strip.set_color("RED")
            tts_to_play = os.path.join(TTS_DIR, "color_red.wav")

        elif label == "COLOR_BLUE":
            device_state["light_color"] = "BLUE"
            self.strip.set_color("BLUE")
            tts_to_play = os.path.join(TTS_DIR, "color_blue.wav")

        elif label == "COLOR_GREEN":
            device_state["light_color"] = "GREEN"
            self.strip.set_color("GREEN")
            tts_to_play = os.path.join(TTS_DIR, "color_green.wav")

        # --- MUSIC ---
        elif label == "PLAY_MUSIC":
            device_state["music"] = "PLAYING"
            self.audio.play_music()
            tts_to_play = os.path.join(TTS_DIR, "music_play.wav")

        elif label == "PAUSE":
            device_state["music"] = "PAUSED"
            self.audio.pause_music()
            tts_to_play = os.path.join(TTS_DIR, "music_pause.wav")

        elif label == "STOP":
            device_state["music"] = "STOPPED"
            self.audio.stop_music()
            tts_to_play = os.path.join(TTS_DIR, "music_stop.wav")

        elif label == "NEXT":
            self.audio.next_track()
            tts_to_play = os.path.join(TTS_DIR, "music_next.wav")

        elif label == "VOLUME_UP":
            self.audio.volume_up(10)
            device_state["volume"] = self.audio.music_volume
            tts_to_play = os.path.join(TTS_DIR, "vol_up.wav")

        elif label == "VOLUME_DOWN":
            self.audio.volume_down(10)
            device_state["volume"] = self.audio.music_volume
            tts_to_play = os.path.join(TTS_DIR, "vol_down.wav")

        # --- THERMOSTAT / TEMPERATURE (Set point command) ---
        elif label == "TEMPERATURE_18":
            device_state["thermostat"] = "18°C"
            self.tm1637.show_temperature_popup(18, duration=5.0)
            tts_to_play = os.path.join(TTS_DIR, "temp_set_18.wav")

        elif label == "TEMPERATURE_22":
            device_state["thermostat"] = "22°C"
            self.tm1637.show_temperature_popup(22, duration=5.0)
            tts_to_play = os.path.join(TTS_DIR, "temp_set_22.wav")

        elif label == "TEMPERATURE_26":
            device_state["thermostat"] = "26°C"
            self.tm1637.show_temperature_popup(26, duration=5.0)
            tts_to_play = os.path.join(TTS_DIR, "temp_set_26.wav")

        # --- TIME (Composed from pre-made clips via RTC) ---
        elif label == "TIME":
            now_dt = self.rtc.read_time()
            hour_12 = now_dt.hour % 12
            if hour_12 == 0:
                hour_12 = 12
            minute = now_dt.minute
            ampm = "am" if now_dt.hour < 12 else "pm"

            tts_sequence = [
                os.path.join(TTS_DIR, "the_time_is.wav"),
                os.path.join(TTS_DIR, f"num_{hour_12}.wav")
            ]
            if minute > 0:
                tts_sequence.append(os.path.join(TTS_DIR, f"num_{minute}.wav"))
            tts_sequence.append(os.path.join(TTS_DIR, f"{ampm}.wav"))

        # --- WEATHER (DHT22 reading + rule-based classification) ---
        elif label == "WEATHER":
            temp_c, humidity = self.dht.read_readings(retries=3)
            if temp_c is not None and humidity is not None:
                cond_code, cond_str = self.dht.evaluate_weather_condition(temp_c, humidity)
                t_int = int(round(temp_c))
                h_int = int(round(humidity))

                print(f"⛅ [WEATHER]: {t_int}°C, {h_int}% Humidity -> {cond_str}")
                tts_sequence = [
                    os.path.join(TTS_DIR, "the_weather_is.wav"),
                    os.path.join(TTS_DIR, f"cond_{cond_code}.wav"),
                    os.path.join(TTS_DIR, "temperature_is.wav"),
                    os.path.join(TTS_DIR, f"num_{t_int}.wav"),
                    os.path.join(TTS_DIR, "degrees.wav"),
                    os.path.join(TTS_DIR, "humidity_is.wav"),
                    os.path.join(TTS_DIR, f"num_{h_int}.wav"),
                    os.path.join(TTS_DIR, "percent.wav")
                ]
            else:
                print("⛅ [WEATHER ERROR]: DHT22 failed to read after retries.")
                tts_to_play = os.path.join(TTS_DIR, "dht_error.wav")

        # --- TIMER ---
        elif label == "TIMER_10s":
            device_state["timer"] = "10 seconds"
            self.alarm_timer.start_timer(10)
            tts_to_play = os.path.join(TTS_DIR, "timer_10s.wav")

        elif label == "TIMER_30s":
            device_state["timer"] = "30 seconds"
            self.alarm_timer.start_timer(30)
            tts_to_play = os.path.join(TTS_DIR, "timer_30s.wav")

        elif label == "TIMER_1m":
            device_state["timer"] = "1 minute"
            self.alarm_timer.start_timer(60)
            tts_to_play = os.path.join(TTS_DIR, "timer_1m.wav")

        # --- ALARM ---
        elif label == "ALARM_6_00AM":
            device_state["alarm"] = "6:00 AM"
            self.state_mgr.set_alarm("6:00 AM", 6, 0)
            tts_to_play = os.path.join(TTS_DIR, "alarm_6am.wav")

        elif label == "ALARM_8_00AM":
            device_state["alarm"] = "8:00 AM"
            self.state_mgr.set_alarm("8:00 AM", 8, 0)
            tts_to_play = os.path.join(TTS_DIR, "alarm_8am.wav")

        elif label == "ALARM_9_00PM":
            device_state["alarm"] = "9:00 PM"
            self.state_mgr.set_alarm("9:00 PM", 21, 0)
            tts_to_play = os.path.join(TTS_DIR, "alarm_9pm.wav")

        # --- CREATE REMINDER ---
        elif label == "CREATE_REMINDER_DRINK_WATER":
            already = self.state_mgr.add_or_refresh_reminder("Drink Water")
            device_state["reminders"] = self.state_mgr.list_reminders()
            tts_to_play = os.path.join(TTS_DIR, "reminder_exists.wav" if already else "reminder_added.wav")

        elif label == "CREATE_REMINDER_STUDY":
            already = self.state_mgr.add_or_refresh_reminder("Study")
            device_state["reminders"] = self.state_mgr.list_reminders()
            tts_to_play = os.path.join(TTS_DIR, "reminder_exists.wav" if already else "reminder_added.wav")

        elif label == "CREATE_REMINDER_EXERCISE":
            already = self.state_mgr.add_or_refresh_reminder("Exercise")
            device_state["reminders"] = self.state_mgr.list_reminders()
            tts_to_play = os.path.join(TTS_DIR, "reminder_exists.wav" if already else "reminder_added.wav")

        # --- LIST REMINDERS ---
        elif label == "LIST_REMINDERS":
            active_rems = self.state_mgr.list_reminders()
            device_state["reminders"] = active_rems
            if not active_rems:
                tts_to_play = os.path.join(TTS_DIR, "no_reminders.wav")
            else:
                tts_sequence = [os.path.join(TTS_DIR, "your_reminders_are.wav")]
                for r in active_rems:
                    r_clean = r.lower().replace(" ", "_")
                    clip = os.path.join(TTS_DIR, f"rem_{r_clean}.wav")
                    tts_sequence.append(clip)

        # --- TELEPHONY ---
        elif label == "CALL":
            self.telephony.make_call_async()
            tts_to_play = os.path.join(TTS_DIR, "calling.wav")

        elif label == "MESSAGE":
            self.telephony.send_sms_async()
            tts_to_play = os.path.join(TTS_DIR, "sending_message.wav")

        # Play TTS reply (wait until finished before resuming wake word)
        if tts_sequence:
            self.audio.play_tts_sequence(tts_sequence, wait=True)
        elif tts_to_play and os.path.exists(tts_to_play):
            self.audio.play_sfx(tts_to_play, wait=True)

        return device_state

    def cleanup(self):
        """Clean shutdown of all hardware."""
        print("\n[HARDWARE MANAGER]: Cleaning up GPIO, displays, and audio streams...")
        try:
            self.audio.close()
        except Exception:
            pass
        try:
            self.status_led.close()
        except Exception:
            pass
        try:
            self.strip.close()
        except Exception:
            pass
        try:
            self.lcd.close()
        except Exception:
            pass
        try:
            self.tm1637.close()
        except Exception:
            pass
        try:
            self.rtc.close()
        except Exception:
            pass
        try:
            self.dht.close()
        except Exception:
            pass
        try:
            self.alarm_timer.close()
        except Exception:
            pass
        print("[HARDWARE MANAGER]: Shutdown complete.")
