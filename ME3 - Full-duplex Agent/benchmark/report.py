#!/usr/bin/env python3
"""
Benchmark Latency Reporter
Parses benchmark_turns.csv and reports:
- End-of-speech to first-audio-played latency (Median & p95)
- Tool-call overhead (counts, mean, median, p95)
- Barge-in stop time (Median & p95)
"""

import sys
import csv
import numpy as np
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import BENCHMARK_LOG_PATH


def parse_float(val):
    try:
        return float(val) if val not in ("", None) else None
    except ValueError:
        return None


def calculate_stats(values):
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def generate_benchmark_report(csv_path=BENCHMARK_LOG_PATH):
    p = Path(csv_path)
    if not p.exists():
        print(f"❌ Benchmark CSV not found at {p}. Run the assistant or turns first.")
        return

    turns = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turns.append(row)

    if not turns:
        print("Benchmark CSV is empty.")
        return

    eos_to_speaker = []
    eos_to_model = []
    speaker_buffer = []
    tool_durations = {}
    barge_in_stops = []

    for r in turns:
        e2s = parse_float(r.get("eos_to_speaker_audio_ms"))
        if e2s is not None and e2s > 0:
            eos_to_speaker.append(e2s)

        e2m = parse_float(r.get("eos_to_model_audio_ms"))
        if e2m is not None and e2m > 0:
            eos_to_model.append(e2m)

        sb = parse_float(r.get("speaker_buffer_ms"))
        if sb is not None and sb > 0:
            speaker_buffer.append(sb)

        tool_name = r.get("tool_name")
        tdur = parse_float(r.get("tool_duration_ms"))
        if tool_name and tdur is not None and tdur > 0:
            if tool_name not in tool_durations:
                tool_durations[tool_name] = []
            tool_durations[tool_name].append(tdur)

        bstop = parse_float(r.get("barge_in_stop_ms"))
        if bstop is not None and bstop > 0:
            barge_in_stops.append(bstop)

    s_e2s = calculate_stats(eos_to_speaker)
    s_e2m = calculate_stats(eos_to_model)
    s_sb = calculate_stats(speaker_buffer)
    s_bstop = calculate_stats(barge_in_stops)

    print("\n" + "=" * 70)
    print("AI 231 ME3: FULL-DUPLEX VOICE AGENT LATENCY BENCHMARK REPORT")
    print("=" * 70)
    print(f"Total Turn Records: {len(turns)}")
    print(f"Benchmark File:     {p}\n")

    print("1. END-OF-SPEECH TO FIRST-AUDIO-PLAYED LATENCY:")
    print(f"   - Samples Recorded: {s_e2s['count']}")
    print(f"   - Median (p50):     {s_e2s['median']:.1f} ms")
    print(f"   - 95th %ile (p95):  {s_e2s['p95']:.1f} ms")
    print(f"   - Min / Max:        {s_e2s['min']:.1f} ms / {s_e2s['max']:.1f} ms")
    print(f"   - Mean:             {s_e2s['mean']:.1f} ms")

    print("\n2. NETWORK & SERVER TTFT (EOS to First Cloud Audio Chunk):")
    print(f"   - Median (p50):     {s_e2m['median']:.1f} ms")
    print(f"   - 95th %ile (p95):  {s_e2m['p95']:.1f} ms")

    print("\n3. LOCAL SPEAKER BUFFERING / RESAMPLING LATENCY:")
    print(f"   - Median (p50):     {s_sb['median']:.1f} ms")
    print(f"   - 95th %ile (p95):  {s_sb['p95']:.1f} ms")

    print("\n4. BARGE-IN REACTION TIME (User Speech Onset to Playback Silent):")
    print(f"   - Barge-in Events:  {s_bstop['count']}")
    print(f"   - Median (p50):     {s_bstop['median']:.1f} ms")
    print(f"   - 95th %ile (p95):  {s_bstop['p95']:.1f} ms")

    print("\n5. TOOL-CALL OVERHEAD BREAKDOWN:")
    if not tool_durations:
        print("   - No tool executions recorded yet.")
    else:
        print(f"   {'Tool Name':<28} | {'Count':<6} | {'Median':<10} | {'p95':<10} | {'Mean':<10}")
        print("   " + "-" * 68)
        for tname, durations in tool_durations.items():
            ts = calculate_stats(durations)
            print(f"   {tname:<28} | {ts['count']:<6} | {ts['median']:<8.1f}ms | {ts['p95']:<8.1f}ms | {ts['mean']:<8.1f}ms")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_benchmark_report()

