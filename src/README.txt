# Spatially Aware Smart Glasses — Repaired Pilot Pipeline

This version follows the FYP plan: one Qwen Realtime session per recording; an instruction supplied once; causal, chronological 2 FPS observations (one frame every 0.5 seconds); one `WAIT` or `TRIGGER: <reminder>` response after each frame; first-trigger one-shot stopping; and Precision / Recall / F1 / mean trigger latency.

## Files

- `src/evaluate_realtime.py`: record & replay first-person videos through Qwen Realtime WebSocket.
- `src/evaluate_metrics.py`: score outputs against human annotations.
- `test_pipeline.py`: offline logic tests, no API key required.
- `dataset/annotations.example.jsonl`: example annotations; copy it to `dataset/annotations.jsonl` and change the actual video timestamps.
- `.env.example`: configuration template; rename to `.env` and add your credentials.

## Install (Python 3.10–3.12)

From the directory containing this guide:

```powershell
python -m pip install -r requirements.txt
python test_pipeline.py
```

Copy `.env.example` to `.env` and fill the `DASHSCOPE_API_KEY` and `DASHSCOPE_WORKSPACE_ID` for the **same China (Beijing)** Model Studio workspace. Never share `.env` or API keys.

## Single-video pilot

Place an actual video at `dataset/videos/video_001.mp4`, then:

```powershell
python src/evaluate_realtime.py --video dataset/videos/video_001.mp4 --video-id video_001 --instruction "When you see a restroom sign, remind me that there is a restroom nearby." --output results/pilot_predictions.jsonl
```

The default `--instruction-mode session` puts the sample instruction once in `session.update.instructions` next to the fixed monitor prompt. The original `conversation.item.create` user-text path is available as `--instruction-mode user-item`, **experimental** until confirmed to work with the chosen Qwen model/API. Don't combine results from the two modes in the same main benchmark; document which one you used. Changing the session instruction means each session's system text is sample-specific, even though the monitoring rules stay identical.

To run batch videos, copy the example annotations file, fill it with real annotations, then:

```powershell
python src/evaluate_realtime.py --annotations dataset/annotations.jsonl --video-dir dataset/videos --output results/pilot_predictions.jsonl
```

Additional options: `--fps 2` (default), `--timeout 90`, `--no-realtime-pacing` (process sequential frames without waiting for original timestamps), and `--instruction-mode user-item`.

**Clean rerun**: the real-time program APPENDS records. Use a fresh predictions filename for each run. The scorer warns and uses the last result for duplicate IDs; do not intentionally merge replicates.

## Score the recordings

```powershell
python src/evaluate_metrics.py --annotations dataset/annotations.jsonl --predictions results/pilot_predictions.jsonl --output results/metrics_summary.json --csv results/per_video_metrics.csv --report results/evaluation_report.md
```

At 2 FPS, timestamps normally follow `0.0, 0.5, 1.0, 1.5, ...` seconds. Review the extracted sampled frames when annotating; a target that appears at 10.3 seconds in the original video may first be observable at 10.5 seconds.

For positive videos, annotate the FIRST **sampled** frame where the condition is clearly satisfied (not the first unsampled original video frame). Use `trigger_end_time` for the last sampled time the condition remains satisfied. If the goal never occurs, set `positive:false`, both times `null`.

A positive video triggered within `[trigger_start_time, trigger_end_time]` is TP. An early or after-end one-shot trigger counts as **both FP and FN**, because it is an invalid alarm and the valid one-shot task is missed. A positive video with no trigger is FN; negative + trigger is FP; negative + no trigger is TN. Latency (seconds) is sampled-video trigger timestamp minus sampled-video GT start, **on true positives only**. `INVALID` responses are recorded separately, not treated as valid `WAIT`; check the protocol-quality count. The metrics measure trigger detection/timing; compare `expected_action` to `trigger_message` manually to evaluate reminder-content accuracy.

Failed or missing prediction records are excluded from the headline performance metrics and shown in the summary; review excluded records before reporting performance. Because latency is measured on sampled-video timestamps, it does **not** equal end-to-end real-world user-notification time. The program separately logs model-response and elapsed wall-clock time.

## Important checks before collecting 30–40 videos

1. Run five pilot recordings (landmark positive, negative, distractor, action, compositional).
2. Check a real server returns `session.updated`, `input_audio_buffer.committed`, `response.done`, and `response.status=completed`.
3. Check your chosen instruction mode actually preserves the pending instruction across frames.
4. Inspect predictions and verify sampling timestamps visually before annotating all recordings.
5. Keep model settings and FPS fixed across your primary benchmark.

**External validation status:** Offline unit tests for protocol-event handling and scoring pass. Live Alibaba API connection, available quota, and genuine recognition quality still need validation with your own account and G1 recordings.

Official Qwen docs: https://help.aliyun.com/zh/model-studio/client-events and https://help.aliyun.com/zh/model-studio/omni-realtime-interaction-process .

## Deviation from the original FYP plan

The advisor's initial written plan specifies approximately **1 FPS**. This revision intentionally uses **2 FPS** (one image every 0.5 seconds), which can improve capture of brief actions but increases frame/API requests and may increase cost. State the change explicitly in your methodology and obtain approval if the experimental setting must match the plan exactly. The system remains causal/sequential; frame-by-frame API waiting can cause real wall-clock delivery to fall behind a true 2 FPS stream.
