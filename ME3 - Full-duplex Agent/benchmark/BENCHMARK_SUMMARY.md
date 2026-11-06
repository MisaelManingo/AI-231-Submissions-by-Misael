# Berry full-duplex benchmark summary

Student 2020-11106 · run 2026-11-06 21:44 (PHT) · Raspberry Pi 5 (8 GB) · agent = Gemini Live speech-to-speech
(no separate ASR/LLM/TTS stages).

Tool: course harness `fd-benchmark` (Full-Duplex-Bench v1/v1.5 + missed/false turn-taking), direct feed, 1 round,
no LLM judge, stage-latency phrase test skipped. Mock hardware; web search, image/sound generation, Spotify and live
captions disabled via env flags. Attention gate, VAD and barge-in at defaults. The benchmark audio replaced the USB
mic/speaker through a benchmark-only adapter (no agent source changed), so ALSA I/O and the acoustic echo path were
not exercised.

## 1. fdbench results (1 round, n = 1-2 clips per test: indicative only)

| Metric | Result |
|---|---|
| Missed turn-taking rate | 100% (1/1) |
| False turn-taking rate | 50% (1/2): side talk 0% (0/1), TV/radio 100% (1/1) |
| TOR pause handling (synthetic, Candor) | 0% (0/1 each) |
| TOR interruption | 100% (1/1) |
| Barge-in stop time (speech start to playback stopped) | 1.61 s mean, 1.74 s p95 (n=2); target < 0.80 s |
| v1.5 timing shares | backchannel / talking-to-others: kept talking 100%; background speech: stopped 100% (stop latency 1.28 s, n=1) |
| Machine | peak 56 °C, never throttled, CPU median 5%, agent RAM peak ~624 MB |

Not applicable to a speech-to-speech agent: ASR-final, LLM-TTFT and TTS-TTFB stage latencies and WER (no separate
stages or ASR transcript). Judge-based scores were not run.

## 2. End-to-end latency from `benchmark_turns.csv` (end of speech to first speaker audio)

De-duplicated per turn. The file covers 2026-11-04 to 2026-11-06 of normal Berry use, not only the benchmark run.

| Scope | Turns | Median | p95 | Mean | Max |
|---|---|---|---|---|---|
| Whole file | 365 | 1.76 s | 7.53 s | 2.45 s | 26.4 s |
| Turns without a tool call | 235 | 1.54 s | 6.57 s | 2.10 s | 21.0 s |
| Turns with a tool call | 130 | 2.23 s | 8.27 s | 3.08 s | 26.4 s |
| 2026-11-06 13:xx UTC rows (benchmark hour) | 20 | 2.02 s | 3.03 s | 2.24 s | 5.66 s |

## Caveats

- The benchmark hour includes tool calls (spotify_play, control_media, delegate_long_task, cancel_timer); the
  mock was not fully tool-free.
- The CSV barge-in stop time is 0 ms on all rows: it only times the local playback flush, not speech-start to
  stop, so it is not comparable with the fdbench figure above.
- Single 1-round run: rates have n of 1-2 and are not statistically meaningful.
