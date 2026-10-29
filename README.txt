# Spatially_Aware_Smart_Glasses

A research prototype for the University of Macau CISC4000 FYP **Spatially Aware Multimodal AI Assistant for Smart Glasses**. This project replays recorded first-person videos to evaluate whether Qwen3.8-Omni-Flash-Realtime can retain a pending instruction and trigger a reminder when the corresponding visual condition appears.

## Features

- Creates an independent Realtime API session for each video.
- Sets a fixed monitoring prompt and sends the pending instruction once as a separate user text message.
- Samples and replays video frames in chronological order, using 1 FPS by default.
- Requests one `WAIT` / `TRIGGER` decision after each new frame.
- Stops the one-shot evaluation after the first `TRIGGER`.
- Saves per-frame predictions, raw API events, response times, and reported token-usage counters to JSONL.
- Supports single-video runs and batch runs from a JSONL annotation file.

## Project Structure

```text
project/
├── .env
├── README.md
├── requirements.txt
├── src/
│   └── evaluate_realtime.py
├── dataset/
│   ├── annotations.jsonl         # Create locally; review before committing
│   └── videos/                   # Local video files; excluded from Git
└── results/                      # Experiment outputs; excluded from Git
```

## Requirements

- Windows 10/11 (64-bit recommended).
- Python 3.10–3.12 recommended.
- An Alibaba Cloud Model Studio China workspace with the matching API key and Workspace ID.
- A local recorded video, for example `dataset/videos/test.mp4`.

## Setup

### 1. Configure environment variables

Copy `.env.example` to `.env` in the project root and fill in your own credentials:

```dotenv
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_WORKSPACE_ID=your_beijing_workspace_id
```

Use the API key and Workspace ID belonging to the same Model Studio workspace/region. The exact region, model availability, quota, and endpoint configuration should be verified in the Alibaba Cloud console before running experiments.

### 2. Install dependencies

Open a VS Code terminal in the project root. Install dependencies in the same Python environment that you will use to run the script:

```powershell
python -m pip install -r requirements.txt
```

Verify the main dependencies:

```powershell
python -c "import cv2, websocket; print('OpenCV:', cv2.__version__); print('websocket-client: OK')"
```

If pip tries to build OpenCV from source on Windows, use a compatible prebuilt wheel, for example:

```powershell
python -m pip install --only-binary=:all: "opencv-python==4.11.0.86"
```

### 3. Prepare a video

Place a video you are authorized to use in `dataset/videos/`, for example:

```text
dataset/videos/test.mp4
```

## Running the Program

Run all commands from the project root, for example `D:\CISC4000\project`. Ensure the script and video paths match your actual folder layout.

### A. Evaluate one video

```powershell
python src\evaluate_realtime.py --video dataset\videos\test.mp4 --instruction "When you see a restroom sign, remind me that there is a restroom nearby." --output results\qwen38_realtime_predictions_test.jsonl
```

Arguments:

- `--video`: Path to a local video file.
- `--instruction`: The pending instruction, sent once at the beginning of the session.
- `--output`: JSONL file to which the result record will be appended.

Optional labels can be supplied after the ground-truth trigger time has been manually verified from the sampled frames:

```powershell
python src\evaluate_realtime.py --video dataset\videos\test.mp4 --instruction "When you see a restroom sign, remind me that there is a restroom nearby." --positive true --ground-truth-trigger-time 10 --expected-action "Remind the user that a restroom store is nearby." --output results\qwen38_realtime_predictions_test.jsonl
```

The value `10` is only an example. Do not use it unless the video has been inspected and the first clearly identifiable sampled frame has been confirmed as 10 seconds.


### Useful optional arguments

```text
--fps 1.0                     Target sampling/replay rate; default is 1 FPS
--timeout 90                  Timeout for API events, in seconds
--no-realtime-pacing          Process the next frame as soon as the previous decision returns
--video-id video_001          Set a sample ID in single-video mode
--positive true|false         Label a video as positive or negative
--ground-truth-trigger-time 27 Ground-truth trigger time, in seconds
--expected-action "..."       Expected reminder/action
```

## Output

The output is JSONL. Use a new output filename for each experimental run, or move/delete the previous file before a clean rerun. The program appends records rather than overwriting the file.

Each record includes fields such as:

- `video_id`, `video_path`, `model`, and `fps`;
- `instruction`, `positive`, and `ground_truth_trigger_time`;
- `predictions`: timestamp, raw model text, parsed decision, response status, response time, and encoded image size for each sampled frame;
- `predicted_trigger_time` and `trigger_latency`;
- `raw_api_events`: raw server events for debugging;
- `reported_usage_sum_input_tokens`, `reported_usage_sum_output_tokens`, and `reported_usage_sum_total_tokens`;
- `status` and `error`.

The `reported_usage_sum_*` fields sum the per-response usage counters. Realtime responses may count historical context again on later turns, so these sums are not equivalent to deduplicated unique input tokens. Check actual usage and quota in the Alibaba Cloud console before scaling up the experiment.
