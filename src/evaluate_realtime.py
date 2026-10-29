from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: opencv-python. Install with: pip install opencv-python") from exc

try:
    import websocket
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: websocket-client. Install with: pip install websocket-client") from exc

# Optional: load .env if python-dotenv has been installed; a minimal fallback is included.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value

MODEL_ID = "qwen3.8-omni-flash-realtime"
SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
MAX_BASE64_IMAGE_BYTES = 250 * 1024  # stay safely below the documented 256 KB limit

MONITORING_PROMPT = """You are evaluating an event-triggered prospective instruction-following task from a first-person video.
The user will provide one pending instruction as a user text message at the start of this session. Retain that instruction while observing subsequent frames; do not ask the user to repeat it.
For each new visual observation, make exactly one monitoring decision:
- If the instruction's visual condition is not yet clearly satisfied, output exactly: WAIT
- If the condition is clearly satisfied, output: TRIGGER: <short reminder/action>
Do not trigger early. Do not trigger merely because the instruction exists. Do not trigger on visually or semantically similar distractors that do not satisfy the condition. Use only the images received so far; never assume what a future frame will show. If evidence is insufficient or ambiguous, output WAIT. Do not explain your reasoning. After the first TRIGGER, the one-shot task is complete."""


def load_annotation_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Each JSONL line must be an object: {path}:{line_no}")
            records.append(item)
    return records


def resolve_video_path(sample: dict[str, Any], video_dir: Path) -> Path:
    for key in ("video_path", "video_file", "filename"):
        candidate_value = sample.get(key)
        if candidate_value:
            candidate = Path(str(candidate_value))
            candidates = [candidate] if candidate.is_absolute() else [PROJECT_ROOT / candidate, video_dir / candidate]
            for item in candidates:
                if item.is_file():
                    return item.resolve()

    video_id = str(sample.get("video_id", "")).strip()
    if not video_id:
        raise ValueError("Annotation needs video_id or video_path/video_file/filename")
    direct = video_dir / video_id
    if direct.is_file():
        return direct.resolve()
    for ext in SUPPORTED_VIDEO_EXTS:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.is_file():
            return candidate.resolve()
    matches = [p for p in video_dir.glob(f"{video_id}.*") if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"Could not find a video for video_id={video_id!r} in {video_dir}. "
        f"Expected {video_id} plus one of {', '.join(SUPPORTED_VIDEO_EXTS)}."
    )


def encode_frame_jpeg(frame) -> str:
    """Encode a BGR OpenCV frame as Base64 JPEG below the API size limit."""
    image = frame
    height, width = image.shape[:2]
    max_width, max_height = 1280, 720
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)

    quality = 85
    for _ in range(12):
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("OpenCV failed to encode a video frame as JPEG")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        if len(b64.encode("ascii")) <= MAX_BASE64_IMAGE_BYTES:
            return b64
        if quality > 45:
            quality -= 10
        else:
            h, w = image.shape[:2]
            image = cv2.resize(image, (max(1, int(w * 0.85)), max(1, int(h * 0.85))), interpolation=cv2.INTER_AREA)
            quality = 70
    raise ValueError("Could not reduce frame to below 250 KB Base64. Try lowering the video resolution.")


def iter_video_frames(video_path: Path, target_fps: float = 1.0) -> Iterable[tuple[float, Any]]:
    """Yield one frame per target interval, timestamped in original-video seconds."""
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / source_fps if source_fps > 0 and frame_count > 0 else None
        if duration is not None:
            total_steps = max(1, int(duration * target_fps) + 1)
            for step in range(total_steps):
                timestamp = step / target_fps
                if timestamp >= duration + (0.5 / max(source_fps, 1.0)):
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    continue
                yield timestamp, frame
        else:
            # Fallback for video files where FPS/frame-count metadata is missing.
            next_timestamp = 0.0
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = index / max(source_fps, target_fps, 1.0)
                if timestamp + 1e-6 >= next_timestamp:
                    yield next_timestamp, frame
                    next_timestamp += 1.0 / target_fps
                index += 1
    finally:
        cap.release()


def event_payload(event_type: str, **kwargs: Any) -> str:
    payload = {"type": event_type, **kwargs}
    return json.dumps(payload, ensure_ascii=False)


def _recv_json(ws, timeout_seconds: float = 90.0) -> dict[str, Any]:
    """Receive one JSON event, tolerating periodic websocket read timeouts."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "unparsed_event", "raw": raw}
        if isinstance(data, dict):
            return data
        return {"type": "unparsed_event", "raw": data}
    raise TimeoutError(f"Timed out waiting for a Realtime API event after {timeout_seconds:.0f}s")


def wait_for_event(ws, desired_types: set[str], raw_events: list[dict[str, Any]], timeout_seconds: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        event = _recv_json(ws, max(1.0, deadline - time.monotonic()))
        raw_events.append(event)
        event_type = event.get("type", "")
        if event_type == "error":
            error = event.get("error", {})
            raise RuntimeError(f"Realtime API error: {error.get('code', '')} {error.get('message', json.dumps(error, ensure_ascii=False))}")
        if event_type in desired_types:
            return event
    raise TimeoutError(f"Timed out waiting for event type(s): {', '.join(sorted(desired_types))}")


def extract_response_text(response_done_event: dict[str, Any], frame_events: list[dict[str, Any]]) -> str:
    """Extract final text from official Realtime event shapes."""
    response = response_done_event.get("response", {})
    pieces: list[str] = []
    for output_item in response.get("output", []) or []:
        for content_item in output_item.get("content", []) or []:
            text = content_item.get("text") or content_item.get("transcript")
            if text:
                pieces.append(str(text))
    if pieces:
        return "\n".join(pieces).strip()

    # Fallback to text events if response.done did not include text content.
    for event in frame_events:
        if event.get("type") == "response.text.done" and event.get("text"):
            return str(event["text"]).strip()
    deltas = [str(e.get("delta", "")) for e in frame_events if e.get("type") == "response.text.delta"]
    return "".join(deltas).strip()


def parse_decision(text: str) -> tuple[str, str | None]:
    cleaned = (text or "").strip()
    if re.match(r"^WAIT\b", cleaned, flags=re.IGNORECASE):
        return "WAIT", None
    match = re.match(r"^TRIGGER\s*:?\s*(.*)$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return "TRIGGER", match.group(1).strip() or None
    return "INVALID", None


def build_session_instructions() -> str:
    """Return the fixed monitoring prompt; sample instructions are user messages."""
    return MONITORING_PROMPT


def open_realtime_session(api_key: str, workspace_id: str, pending_instruction: str, timeout_seconds: float) -> tuple[Any, list[dict[str, Any]]]:
    api_url = (
        f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
        f"?model={MODEL_ID}"
    )
    ws = websocket.create_connection(
        api_url,
        header=[f"Authorization: Bearer {api_key}"],
        timeout=10,
        enable_multithread=True,
    )
    ws.settimeout(1.0)
    raw_events: list[dict[str, Any]] = []
    try:
        wait_for_event(ws, {"session.created"}, raw_events, timeout_seconds)
        ws.send(event_payload(
            "session.update",
            session={
                "modalities": ["text"],
                "instructions": build_session_instructions(),
                "turn_detection": None,
                "input_audio_format": "pcm",
                "temperature": 0.1,
                "max_tokens": 128,
            },
        ))
        wait_for_event(ws, {"session.updated"}, raw_events, timeout_seconds)

        # Send the sample-specific pending instruction exactly once as a user text item.
        # Qwen Omni-Realtime supports input_text via conversation.item.create.
        ws.send(event_payload(
            "conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": pending_instruction.strip()}],
            },
        ))
        wait_for_event(ws, {"conversation.item.created"}, raw_events, timeout_seconds)
        return ws, raw_events
    except Exception:
        ws.close()
        raise


def request_monitoring_decision(ws, image_b64: str, timeout_seconds: float) -> tuple[str, str, list[dict[str, Any]], float]:
    """Submit a silent audio placeholder plus one image, then request one text decision."""
    frame_events: list[dict[str, Any]] = []
    start = time.monotonic()

    # Model Studio currently requires an input_audio_buffer.append event before image input.
    # Supply 100 ms of 16 kHz, mono, signed 16-bit PCM silence as a technical placeholder.
    silent_pcm = b"\x00\x00" * 1600
    silence_b64 = base64.b64encode(silent_pcm).decode("ascii")
    ws.send(event_payload("input_audio_buffer.append", audio=silence_b64))
    ws.send(event_payload("input_image_buffer.append", image=image_b64))
    ws.send(event_payload("input_audio_buffer.commit"))
    wait_for_event(ws, {"input_audio_buffer.committed"}, frame_events, timeout_seconds)

    ws.send(event_payload("response.create"))
    response_done = wait_for_event(ws, {"response.done"}, frame_events, timeout_seconds)
    response_text = extract_response_text(response_done, frame_events)
    elapsed = time.monotonic() - start
    return response_text, response_done.get("response", {}).get("status", "unknown"), frame_events, elapsed


def summarize_reported_usage(raw_events: list[dict[str, Any]]) -> dict[str, int]:
    """Sum usage counters reported for each completed response; includes repeated history."""
    totals = {"reported_usage_sum_input_tokens": 0, "reported_usage_sum_output_tokens": 0, "reported_usage_sum_total_tokens": 0}
    for event in raw_events:
        response = event.get("response", {})
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            continue
        for key, source_key in (
            ("reported_usage_sum_input_tokens", "input_tokens"),
            ("reported_usage_sum_output_tokens", "output_tokens"),
            ("reported_usage_sum_total_tokens", "total_tokens"),
        ):
            value = usage.get(source_key, 0)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def save_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_sample(
    sample: dict[str, Any],
    video_path: Path,
    api_key: str,
    workspace_id: str,
    target_fps: float,
    decision_timeout: float,
    replay_paced: bool,
    output_path: Path,
) -> dict[str, Any]:
    video_id = str(sample.get("video_id") or video_path.stem)
    instruction = str(sample.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"Sample {video_id}: missing 'instruction'")

    gt_value = sample.get("trigger_start_time", sample.get("ground_truth_trigger_time"))
    try:
        ground_truth = float(gt_value) if gt_value is not None else None
    except (TypeError, ValueError):
        ground_truth = None

    record: dict[str, Any] = {
        "video_id": video_id,
        "video_path": str(video_path),
        "model": MODEL_ID,
        "fps": target_fps,
        "instruction": instruction,
        "positive": sample.get("positive"),
        "ground_truth_trigger_time": ground_truth,
        "predicted_trigger_time": None,
        "trigger_latency": None,
        "predictions": [],
        "raw_api_events": [],
        "status": "running",
        "error": None,
    }
    session = None
    total_start = time.monotonic()
    try:
        session, setup_events = open_realtime_session(api_key, workspace_id, instruction, decision_timeout)
        record["raw_api_events"].extend({"stage": "session_setup", **event} for event in setup_events)

        replay_start = time.monotonic()
        frames_sent = 0
        for timestamp, frame in iter_video_frames(video_path, target_fps):
            if replay_paced:
                deadline = replay_start + timestamp
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            image_b64 = encode_frame_jpeg(frame)
            model_text, response_status, events, response_wall_time = request_monitoring_decision(
                session, image_b64, decision_timeout
            )
            decision, trigger_message = parse_decision(model_text)
            prediction = {
                "time": round(float(timestamp), 3),
                "decision": decision,
                "model_text": model_text,
                "trigger_message": trigger_message,
                "response_status": response_status,
                "response_wall_time_seconds": round(response_wall_time, 3),
                "image_base64_bytes": len(image_b64.encode("ascii")),
            }
            record["predictions"].append(prediction)
            for event in events:
                record["raw_api_events"].append({"stage": "frame", "video_time": timestamp, **event})
            print(f"[{video_id} @ {timestamp:6.1f}s] {decision}: {model_text[:180]}")
            frames_sent += 1

            if decision == "TRIGGER":
                record["predicted_trigger_time"] = float(timestamp)
                if ground_truth is not None:
                    record["trigger_latency"] = round(float(timestamp) - ground_truth, 3)
                break

        record["frames_sent"] = frames_sent
        record["status"] = "completed" if frames_sent > 0 else "failed"
        if frames_sent == 0:
            record["error"] = "No frames could be read from the video."
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR [{video_id}]: {record['error']}", file=sys.stderr)
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        record["wall_time_seconds"] = round(time.monotonic() - total_start, 3)
        record.update(summarize_reported_usage(record["raw_api_events"]))
        save_jsonl_record(output_path, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal 1-FPS video replay using Qwen3.8-Omni-Flash-Realtime.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Run one video, e.g. dataset/videos/test.mp4")
    group.add_argument("--annotations", help="JSONL annotation file for batch evaluation")
    parser.add_argument("--instruction", help="Pending instruction for --video mode")
    parser.add_argument("--video-id", help="Optional sample ID for --video mode")
    parser.add_argument("--positive", choices=("true", "false"), default=None, help="Optional label for --video mode")
    parser.add_argument("--ground-truth-trigger-time", type=float, default=None, help="Optional ground-truth time in seconds")
    parser.add_argument("--expected-action", default=None, help="Optional expected reminder/action")
    parser.add_argument("--video-dir", default="dataset/videos", help="Directory of videos in --annotations mode")
    parser.add_argument("--fps", type=float, default=1.0, help="Target sample/replay FPS (default: 1.0)")
    parser.add_argument("--timeout", type=float, default=90.0, help="API event timeout in seconds")
    parser.add_argument("--no-realtime-pacing", action="store_true", help="Do not wait for original video timestamps; process each frame as soon as the previous decision returns")
    parser.add_argument("--output", default="results/qwen38_realtime_predictions.jsonl", help="Output JSONL path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    workspace_id = (os.getenv("DASHSCOPE_WORKSPACE_ID") or os.getenv("WORKSPACE_ID") or "").strip()
    if not api_key:
        print("Missing DASHSCOPE_API_KEY. Add it to the project-root .env file.", file=sys.stderr)
        return 2
    if not workspace_id:
        print("Missing DASHSCOPE_WORKSPACE_ID. Add your Beijing-region Model Studio Workspace ID to the project-root .env file.", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if args.video:
        if not args.instruction:
            print("--instruction is required when using --video", file=sys.stderr)
            return 2
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.is_file():
            print(f"Video not found: {video_path}", file=sys.stderr)
            return 2
        sample: dict[str, Any] = {
            "video_id": args.video_id or video_path.stem,
            "instruction": args.instruction,
            "expected_action": args.expected_action,
            "positive": (args.positive == "true") if args.positive is not None else None,
            "trigger_start_time": args.ground_truth_trigger_time,
        }
        records = [(sample, video_path.resolve())]
    else:
        annotation_path = Path(args.annotations)
        if not annotation_path.is_absolute():
            annotation_path = PROJECT_ROOT / annotation_path
        if not annotation_path.is_file():
            print(f"Annotation file not found: {annotation_path}", file=sys.stderr)
            return 2
        video_dir = Path(args.video_dir)
        if not video_dir.is_absolute():
            video_dir = PROJECT_ROOT / video_dir
        records = []
        for sample in load_annotation_file(annotation_path):
            records.append((sample, resolve_video_path(sample, video_dir)))

    print(f"Model: {MODEL_ID}")
    print(f"Workspace ID: {workspace_id}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(records)} | FPS: {args.fps} | realtime pacing: {not args.no_realtime_pacing}")
    success_count = 0
    for sample, video_path in records:
        result = run_sample(
            sample=sample,
            video_path=video_path,
            api_key=api_key,
            workspace_id=workspace_id,
            target_fps=args.fps,
            decision_timeout=args.timeout,
            replay_paced=not args.no_realtime_pacing,
            output_path=output_path,
        )
        if result.get("status") == "completed":
            success_count += 1
        print(f"Finished {result.get('video_id')}: status={result.get('status')}, predicted_trigger_time={result.get('predicted_trigger_time')}")

    print(f"Done: {success_count}/{len(records)} samples completed.")
    if records:
        print("Note: reported_usage_sum_* fields sum the per-response usage counters; historical context is counted again on later responses.")
    print("Note: this script appends result records. Move/delete the old output JSONL before a clean rerun to avoid duplicate video IDs.")
    return 0 if success_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
