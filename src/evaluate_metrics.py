from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            items.append(item)
    return items


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "1", "positive"}:
            return True
        if value in {"false", "no", "0", "negative"}:
            return False
    return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_predicted_trigger(record: dict[str, Any]) -> float | None:
    """Use the summary timestamp, falling back to the first TRIGGER decision."""
    timestamp = parse_float(record.get("predicted_trigger_time"))
    if timestamp is not None:
        return timestamp

    predictions = record.get("predictions")
    if isinstance(predictions, list):
        for item in predictions:
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision", "")).strip().upper()
            if decision == "TRIGGER":
                timestamp = parse_float(item.get("time"))
                if timestamp is not None:
                    return timestamp
    return None


def first_trigger_decision(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first actual TRIGGER decision and its wall-clock telemetry."""
    predictions = record.get("predictions")
    if isinstance(predictions, list):
        for decision in predictions:
            if isinstance(decision, dict) and str(decision.get("decision", "")).strip().upper() == "TRIGGER":
                return decision
    return None


def utc_seconds(value: Any) -> float | None:
    """Parse an explicitly logged ISO-8601 timestamp; never infer it from video time."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None  # Wall timestamps must carry a timezone.
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def measured_end_to_end_seconds(record: dict[str, Any]) -> tuple[float | None, str | None]:
    """Require an actual event timestamp AND actual notification-delivery timestamp.

    These must be captured by the replay/live application using one clock.
    A Qwen response timestamp alone does NOT certify user notification delivery.
    """
    start = parse_float(record.get("event_wall_elapsed_seconds"))
    end = parse_float(record.get("notification_wall_elapsed_seconds"))
    if start is not None and end is not None and 0 <= start <= end:
        return round(end - start, 3), "monotonic_elapsed"

    start = utc_seconds(record.get("event_wall_time_utc"))
    end = utc_seconds(record.get("notification_wall_time_utc"))
    if start is not None and end is not None and 0 <= end - start:
        return round(end - start, 3), "utc_timestamps"
    return None, None


def sample_wall_clock_telemetry(
    annotation: dict[str, Any], prediction: dict[str, Any], predicted: float | None, gt: float | None
) -> dict[str, Any]:
    """Keep model processing time distinct from video time and real E2E time.

    Legacy replay records contain decision_wall_elapsed_seconds measured since the
    run's timer started. Without an explicit replay origin, subtracting the video
    timestamp assumes video t=0 coincides with run timer t=0. This is only a
    replay-alignment ESTIMATE, not a measured live-camera E2E latency.
    """
    decisions = [d for d in prediction.get("predictions", []) if isinstance(d, dict)] if isinstance(prediction.get("predictions"), list) else []
    durations = [v for d in decisions if (v := parse_float(d.get("response_wall_time_seconds"))) is not None and v >= 0]
    trig = first_trigger_decision(prediction)
    response_wall = parse_float(trig.get("response_wall_time_seconds")) if trig else None
    decision_elapsed = parse_float(trig.get("decision_wall_elapsed_seconds")) if trig else None
    trigger_decision_video_time = parse_float(trig.get("time")) if trig else None
    if response_wall is not None and response_wall < 0:
        response_wall = None
    if decision_elapsed is not None and decision_elapsed < 0:
        decision_elapsed = None

    replay_origin = parse_float(prediction.get("replay_start_wall_elapsed_seconds"))
    explicit_origin = replay_origin is not None
    if replay_origin is None:
        replay_origin = 0.0
    replay_lag = None
    if gt is not None and gt >= 0 and decision_elapsed is not None:
        replay_lag = round(decision_elapsed - (replay_origin + gt), 3)

    measured_e2e, e2e_source = measured_end_to_end_seconds(prediction)
    event_elapsed = parse_float(prediction.get("event_wall_elapsed_seconds"))
    measured_event_to_decision = None
    if event_elapsed is not None and event_elapsed >= 0 and decision_elapsed is not None and decision_elapsed >= event_elapsed:
        measured_event_to_decision = round(decision_elapsed - event_elapsed, 3)

    metadata_gt = parse_float(prediction.get("ground_truth_trigger_time"))
    mismatch = None
    if gt is not None and metadata_gt is not None and abs(gt - metadata_gt) > 1e-6:
        mismatch = round(metadata_gt - gt, 3)

    fps = parse_float(prediction.get("fps"))
    mean_duration = mean(durations) if durations else None
    video_trigger_mismatch = None
    if predicted is not None and trigger_decision_video_time is not None and abs(predicted - trigger_decision_video_time) > 1e-6:
        video_trigger_mismatch = round(trigger_decision_video_time - predicted, 3)

    return {
        "model_response_mean_seconds": round(mean_duration, 4) if mean_duration is not None else None,
        "model_response_samples": len(durations),
        "trigger_response_wall_time_seconds": response_wall,
        "trigger_decision_wall_elapsed_seconds": decision_elapsed,
        "estimated_replay_event_to_decision_seconds": replay_lag,
        "replay_lag_origin": "explicit_replay_start" if explicit_origin else "assumed_run_start_equals_video_start",
        "measured_event_to_decision_seconds": measured_event_to_decision,
        "measured_end_to_end_latency_seconds": measured_e2e,
        "measured_e2e_source": e2e_source,
        "fps": fps,
        "average_response_exceeds_frame_interval": bool(
            fps is not None and fps > 0 and mean_duration is not None and mean_duration > 1 / fps
        ),
        "prediction_gt_metadata_mismatch_seconds": mismatch,
        "prediction_trigger_time_mismatch_seconds": video_trigger_mismatch,
    }


def build_prediction_index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Keep the last record for repeated video IDs; report duplicates explicitly."""
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        video_id = str(record.get("video_id", "")).strip()
        if not video_id:
            continue
        if video_id in indexed:
            duplicates.append(video_id)
        indexed[video_id] = record
    return indexed, sorted(set(duplicates))


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tn = sum(row["tn"] for row in rows)
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    # With single-label scoring an early/invalid trigger is FP only, so recall
    # no longer measures how many positive videos were completed correctly.
    # Keep this independent one-shot success rate to avoid a misleading 100% claim.
    positive_samples = sum(bool(r["positive"]) for r in rows)
    valid_trigger_rate = safe_rate(tp, positive_samples)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    latencies = [row["latency_seconds"] for row in rows if row["latency_seconds"] is not None]
    # Keep each category's means weighted by *per-decision* observations.
    response_count = sum(r.get("model_response_samples", 0) for r in rows)
    response_total = sum(
        r["model_response_mean_seconds"] * r["model_response_samples"]
        for r in rows
        if r.get("model_response_mean_seconds") is not None
    )
    triggered = [r for r in rows if r["predicted_trigger_time"] is not None]
    trigger_response = [r["trigger_response_wall_time_seconds"] for r in triggered if r.get("trigger_response_wall_time_seconds") is not None]
    correct_rows = [r for r in rows if r["outcome"] == "true_positive"]
    replay_lags = [r["estimated_replay_event_to_decision_seconds"] for r in correct_rows if r.get("estimated_replay_event_to_decision_seconds") is not None]
    measured_e2e = [r["measured_end_to_end_latency_seconds"] for r in correct_rows if r.get("measured_end_to_end_latency_seconds") is not None]
    measured_event_to_decision = [r["measured_event_to_decision_seconds"] for r in correct_rows if r.get("measured_event_to_decision_seconds") is not None]
    return {
        "samples": len(rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_samples": positive_samples,
        "valid_trigger_rate": valid_trigger_rate,
        "mean_trigger_latency_seconds": mean(latencies) if latencies else None,
        "latency_samples": len(latencies),
        "mean_model_response_latency_seconds": round(response_total / response_count, 4) if response_count else None,
        "model_response_samples": response_count,
        "mean_estimated_replay_event_to_decision_seconds": round(mean(replay_lags), 4) if replay_lags else None,
        "estimated_replay_lag_samples": len(replay_lags),
        "estimated_replay_lag_assumed_origin_samples": sum(
            r.get("replay_lag_origin") == "assumed_run_start_equals_video_start" and
            r.get("estimated_replay_event_to_decision_seconds") is not None for r in correct_rows
        ),
    }


def delay_bucket(gt: float) -> str:
    # Half-open intervals avoid assigning boundary samples to two groups.
    if gt < 10:
        return "short_0_to_10s"
    if gt < 25:
        return "medium_10_to_25s"
    if gt <= 40:
        return "long_25_to_40s"
    return "over_40s"


def classify_sample(annotation: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any] | None:
    video_id = str(annotation.get("video_id", "")).strip()
    positive = parse_bool(annotation.get("positive"))
    if positive is None:
        return None

    gt = parse_float(annotation.get("trigger_start_time", annotation.get("ground_truth_trigger_time")))
    if positive and (gt is None or gt < 0):
        return None
    end = parse_float(annotation.get("trigger_end_time"))
    if positive and end is not None and (end < gt or end < 0):
        return None
    status = str(prediction.get("status", "unknown")).lower()
    if status not in {"completed", "success", "succeeded"}:
        return None

    predicted = first_predicted_trigger(prediction)
    if predicted is not None and predicted < 0:
        return None
    row: dict[str, Any] = {
        "video_id": video_id,
        "trigger_type": str(annotation.get("trigger_type") or "unknown"),
        "positive": positive,
        "ground_truth_trigger_time": gt,
        "trigger_end_time": end,
        "invalid_output_count": sum(1 for p in prediction.get("predictions", []) if isinstance(p, dict) and p.get("decision") == "INVALID"),
        "trigger_message": next((p.get("trigger_message") for p in prediction.get("predictions", []) if isinstance(p, dict) and p.get("decision") == "TRIGGER"), None),
        "expected_action": annotation.get("expected_action"),
        "predicted_trigger_time": predicted,
        "outcome": "",
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "latency_seconds": None,
        "delay_group": delay_bucket(gt) if positive and gt is not None else "not_applicable",
        "prediction_status": status,
        "error": prediction.get("error"),
    }
    row.update(sample_wall_clock_telemetry(annotation, prediction, predicted, gt))

    if not positive:
        if predicted is None:
            row["outcome"] = "true_negative"
            row["tn"] = 1
        else:
            row["outcome"] = "false_positive_on_negative_video"
            row["fp"] = 1
        return row

    # Positive sample.
    if predicted is None:
        row["outcome"] = "false_negative_no_trigger"
        row["fn"] = 1
    elif predicted < gt - 1e-6:
        # Single-error convention: premature TRIGGER counts once as FP only.
        row["outcome"] = "early_trigger"
        row["fp"] = 1
    elif end is not None and predicted > end + 1e-6:
        # Single-error convention: TRIGGER outside the valid window is FP only.
        row["outcome"] = "trigger_after_condition_ended"
        row["fp"] = 1
    else:
        row["outcome"] = "true_positive"
        row["tp"] = 1
        row["latency_seconds"] = round(predicted - gt, 3)
    return row


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(summary: dict[str, Any], per_video: list[dict[str, Any]]) -> str:
    overall = summary["overall"]
    lines = [
        "# Prospective Trigger Evaluation Report",
        "",
        f"- Annotations: `{summary['annotations_file']}`",
        f"- Predictions: `{summary['predictions_file']}`",
        f"- Evaluated samples: **{overall['samples']}**",
        f"- Excluded samples: **{summary['excluded_count']}**",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| TP | {overall['TP']} |",
        f"| FP | {overall['FP']} |",
        f"| FN | {overall['FN']} |",
        f"| TN | {overall['TN']} |",
        f"| Precision | {fmt(overall['precision'])} |",
        f"| Recall | {fmt(overall['recall'])} |",
        f"| F1 | {fmt(overall['f1'])} |",
        f"| Valid trigger rate on positive videos | {fmt(overall['valid_trigger_rate'])} |",
        f"| Mean trigger latency on video timeline (s; TP only) | {fmt(overall['mean_trigger_latency_seconds'])} |",
        f"| Mean model response time per decision (s; all decisions) | {fmt(overall['mean_model_response_latency_seconds'])} |",
        f"| Mean estimated replay event-to-decision lag (s; TP only) | {fmt(overall['mean_estimated_replay_event_to_decision_seconds'])} |",
        "",
        "## Results by Trigger Type",
        "",
        "| Trigger type | N | TP | FP | FN | TN | Precision | Recall | F1 | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, metrics in summary["by_trigger_type"].items():
        lines.append(
            f"| {category} | {metrics['samples']} | {metrics['TP']} | {metrics['FP']} | {metrics['FN']} | {metrics['TN']} | "
            f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | {fmt(metrics['f1'])} | {fmt(metrics['mean_trigger_latency_seconds'])} |"
        )
    lines += [
        "",
        "## Results by Trigger Delay",
        "",
        "| Delay group | N | TP | FP | FN | TN | Precision | Recall | F1 | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, metrics in summary["by_delay"].items():
        lines.append(
            f"| {group} | {metrics['samples']} | {metrics['TP']} | {metrics['FP']} | {metrics['FN']} | {metrics['TN']} | "
            f"{fmt(metrics['precision'])} | {fmt(metrics['recall'])} | {fmt(metrics['f1'])} | {fmt(metrics['mean_trigger_latency_seconds'])} |"
        )
    lines += [
        "",
        "## Per-video Results",
        "",
        "| Video | Type | GT video (s) | Pred video (s) | Outcome | Video latency (s) | Trigger response (s) | Decision wall elapsed (s) | Replay lag estimate (s) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in per_video:
        lines.append(
            f"| {row['video_id']} | {row['trigger_type']} | "
            f"{fmt(row['ground_truth_trigger_time'])} | {fmt(row['predicted_trigger_time'])} | {row['outcome']} | "
            f"{fmt(row['latency_seconds'])} | {fmt(row['trigger_response_wall_time_seconds'])} | "
            f"{fmt(row['trigger_decision_wall_elapsed_seconds'])} | {fmt(row['estimated_replay_event_to_decision_seconds'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate Precision, Recall, F1 and trigger latency from JSONL files.")
    parser.add_argument("--annotations", default="dataset/annotations.jsonl", help="Ground-truth annotation JSONL")
    parser.add_argument("--predictions", default="results/qwen38_realtime_predictions.jsonl", help="Model prediction JSONL")
    parser.add_argument("--output", default="results/metrics_summary.json", help="Output summary JSON")
    parser.add_argument("--csv", default="results/per_video_metrics.csv", help="Output per-video CSV")
    parser.add_argument("--report", default="results/evaluation_report.md", help="Output Markdown report")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    predictions_path = Path(args.predictions)
    annotations = read_jsonl(annotations_path)
    predictions_raw = read_jsonl(predictions_path)
    prediction_index, duplicate_ids = build_prediction_index(predictions_raw)

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_annotation_ids: set[str] = set()
    for annotation in annotations:
        video_id = str(annotation.get("video_id", "")).strip()
        if not video_id:
            excluded.append({"video_id": "", "reason": "annotation_missing_video_id"})
            continue
        if video_id in seen_annotation_ids:
            raise ValueError(f"Duplicate video_id in annotations: {video_id}")
        seen_annotation_ids.add(video_id)

        prediction = prediction_index.get(video_id)
        if prediction is None:
            excluded.append({"video_id": video_id, "reason": "prediction_not_found"})
            continue
        status = str(prediction.get("status", "unknown")).lower()
        if status not in {"completed", "success", "succeeded"}:
            excluded.append({"video_id": video_id, "reason": f"prediction_status_{status}"})
            continue

        row = classify_sample(annotation, prediction)
        if row is None:
            positive = parse_bool(annotation.get("positive"))
            reason = "invalid_or_missing_positive_label"
            if positive is True and parse_float(annotation.get("trigger_start_time", annotation.get("ground_truth_trigger_time"))) is None:
                reason = "positive_missing_ground_truth_trigger_time"
            elif positive is True:
                reason = "invalid_trigger_time_window"
            excluded.append({"video_id": video_id, "reason": reason})
            continue
        rows.append(row)

    overall = calculate_metrics(rows)
    by_category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_delay_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category_rows[row["trigger_type"]].append(row)
        if row["positive"]:
            by_delay_rows[row["delay_group"]].append(row)

    summary = {
        "annotations_file": str(annotations_path),
        "predictions_file": str(predictions_path),
        "model": next((p.get("model") for p in reversed(predictions_raw) if p.get("model")), None),
        "annotation_count": len(annotations),
        "prediction_record_count": len(predictions_raw),
        "evaluated_count": len(rows),
        "excluded_count": len(excluded),
        "excluded_samples": excluded,
        "duplicate_prediction_video_ids_using_latest_record": duplicate_ids,
        "unmatched_prediction_video_ids": sorted(set(prediction_index) - seen_annotation_ids),
        "overall": overall,
        "by_trigger_type": {key: calculate_metrics(value) for key, value in sorted(by_category_rows.items())},
        "by_delay": {key: calculate_metrics(value) for key, value in sorted(by_delay_rows.items())},
    }

    output_path = Path(args.output)
    csv_path = Path(args.csv)
    report_path = Path(args.report)
    for path in (output_path, csv_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "video_id", "trigger_type", "positive", "ground_truth_trigger_time", "trigger_end_time", "expected_action", "trigger_message",
        "predicted_trigger_time", "outcome", "tp", "fp", "fn", "tn", "latency_seconds",
        "delay_group", "prediction_status", "error",
        "fps", "model_response_mean_seconds", "model_response_samples",
        "trigger_response_wall_time_seconds", "trigger_decision_wall_elapsed_seconds",
        "estimated_replay_event_to_decision_seconds", "replay_lag_origin",
                "prediction_gt_metadata_mismatch_seconds",
        "prediction_trigger_time_mismatch_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(markdown_report(summary, rows), encoding="utf-8")

    print("\n=== Prospective Trigger Evaluation ===")
    print(f"Evaluated samples: {overall['samples']} / {len(annotations)}")
    print(f"TP={overall['TP']}  FP={overall['FP']}  FN={overall['FN']}  TN={overall['TN']}")
    print(f"Precision: {fmt(overall['precision'])}")
    print(f"Recall:    {fmt(overall['recall'])}")
    print(f"F1:        {fmt(overall['f1'])}")
    print(f"Valid trigger rate on positive videos: {fmt(overall['valid_trigger_rate'])}")
    print("NOTE: under single-error scoring, Recall may be high because early triggers are FP only.")
    print(f"Mean video-timeline trigger latency (TP only): {fmt(overall['mean_trigger_latency_seconds'])} seconds")
    print(f"Mean model response time (all decisions):      {fmt(overall['mean_model_response_latency_seconds'])} seconds")
    print(f"Mean estimated replay lag (TP only):          {fmt(overall['mean_estimated_replay_event_to_decision_seconds'])} seconds")
    if overall['estimated_replay_lag_assumed_origin_samples']:
        print("NOTE: legacy replay-lag values assume video t=0 matches elapsed run t=0; not true E2E latency.")
    mismatches = [r for r in rows if r['prediction_gt_metadata_mismatch_seconds'] is not None]
    if mismatches:
        print("WARNING: annotation GT differs from prediction metadata for: " + ", ".join(r['video_id'] for r in mismatches))
    if excluded:
        print(f"Excluded samples: {len(excluded)} (see summary JSON for reasons)")
    if duplicate_ids:
        print(f"WARNING: duplicate prediction IDs found; using the latest record: {', '.join(duplicate_ids)}")
    print("\nSaved:")
    print(f"  JSON:     {output_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
