# Brief directory structure & how to find the final highlights

This repository contains the Soccer Highlight Pro Streamlit app and pipeline that processes full-match videos to produce highlight reels and match statistics.

This README explains where inputs, intermediate files, and final outputs live, how exported highlights are organized, and quick commands to re-export or inspect results.

## Key folders

- `uploads/` — Per-task workspace. Each processing run is kept in its own subfolder named by the file hash / task id. Example:

  `uploads/697765c1578450d748fd2ba3f73a83eb5d24727c1b3ee8dd7e5dcefcba57829f/`

  Typical contents of a task folder:
  - `original_video.mp4` (the uploaded source video)
  - `transcript.txt` (optional)
  - `events.json` (detected events: goals, cards, corners, etc.)
  - `statistics.json` (match/team/player stats used by the dashboard)
  - `status.json` (pipeline progress & metadata; contains `final_summary_filename` when complete)
  - `clips/` (directory with per-event short clips, often organized by event type)
  - `summary_*.mp4` and `summary_custom_*.mp4` (category-based and custom reels)
  - `summary_chronological.mp4` (the main chronological highlights reel; file name may vary and is recorded in `status.json`)
  - `exported_to.txt` (marker file written by the app after export; contains the export path)

- `Highlight_outputs/` — Persistent archive of exported highlights. The app copies final outputs here once the pipeline finishes.

  Each run gets its own subfolder named using the match label and run timestamp:

  `<MatchLabel>_YYYYMMDD_HHMMSS/`

  Example: `Highlight_outputs/Portugal_vs_Spain_20251024_153045/`

  Contains:
  - Copies of the chronological summary and category summaries (`summary_*.mp4`)
  - Copies of any `summary_custom_*.mp4`
  - Copies of clips from `clips/` (if present)
  - (Optional) a small README or marker may be present showing the source `uploads/<task_id>`

## How export works (summary)

- When a run completes (`uploads/<task_id>/status.json` contains `"status": "Complete"`), `streamlit_app.py` will attempt to export the final outputs.
- It derives the folder name from `statistics.json` team names (cleaned to alphanumeric + underscores). If team names are not available it falls back to the task id.
- The export folder is named `<MatchLabel>_YYYYMMDD_HHMMSS` and created under `Highlight_outputs/`.
- The app writes `uploads/<task_id>/exported_to.txt` with the export folder path. If this marker exists, the app will not export again.

## Finding the Portugal vs Spain run (example)

If you know the task id (example):

1. Inspect `uploads/697765c1578450d748fd2ba3f73a83eb5d24727c1b3ee8dd7e5dcefcba57829f/status.json` to confirm the run completed and to see the `final_summary_filename`.
2. If `uploads/.../exported_to.txt` exists, open it to get the export folder path (e.g. `Highlight_outputs/Portugal_vs_Spain_20251024_153045`).
3. Otherwise, look under `Highlight_outputs/` for a folder starting with `Portugal_vs_Spain_` and a timestamp.

## Customization

- To change where exported highlights are stored, update the `FINAL_HIGHLIGHT_FOLDER` constant at the top of `streamlit_app.py`.
- To add an explicit "Re-export" button, I can add a safe UI control that copies again even when `exported_to.txt` exists.
