### `README.md`

# Soccer Highlight Pro ⚽

An AI-powered web application that automatically analyzes full soccer match videos to generate compelling highlight reels, detailed statistical dashboards, and custom video clips.

## Overview

This tool streamlines the video editing process by leveraging AI to do the heavy lifting. Simply upload a full match video, and the pipeline will:

1.  Extract the audio from the video.
2.  Transcribe the entire match commentary using OpenAI's Whisper.
3.  Analyze the transcript with Google's Gemini AI to identify key events like goals, fouls, and missed chances.
4.  Generate a detailed statistical summary of the match.
5.  Automatically create individual video clips for each event with descriptive overlays.
6.  Stitch the clips together into a final chronological highlight reel and separate category-based reels (e.g., all goals).
7.  Present the results in an interactive web interface where you can view stats, download videos, and even create your own custom highlight reels.

---

## Features

-   **End-to-End Automation:** From a single video file to a full set of highlights and stats with one click.
-   **AI-Powered Event Detection:** Uses Google Gemini with a sophisticated one-shot prompt to accurately identify and timestamp key match events.
-   **High-Quality Transcription:** Employs OpenAI's Whisper for accurate speech-to-text conversion of match commentary.
-   **Interactive Statistics Dashboard:** Visualizes key match stats, including team performance, goal scorers, and cards issued.
-   **Custom Highlight Creator:** An interactive tool to select specific events and generate a personalized highlight reel.
-   **Multiple Video Outputs:** Generates a main chronological summary, plus separate videos for each event category (all goals, all fouls, etc.).
-   **Intelligent Caching:** Hashes input files to avoid reprocessing videos that have already been analyzed, saving significant time.

---

## System Architecture

The application follows a multi-stage pipeline to process each video:

`Video Upload` -> `Audio Extraction (ffmpeg)` -> `Transcription (Whisper)` -> `AI Analysis (Gemini)` -> `Event & Stats Generation` -> `Clip Creation (ffmpeg)` -> `Final Video Stitching (ffmpeg)` -> `Display in UI (Streamlit)`

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2.  **FFmpeg**: This is a critical dependency for all video and audio processing.
    -   **Windows**: Download the binary from the [official website](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
    -   **macOS (using Homebrew)**: `brew install ffmpeg`
    -   **Linux (using apt)**: `sudo apt update && sudo apt install ffmpeg`
3.  **Google Gemini API Key**:
    -   You need an API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Setup and Installation

Follow these steps carefully to get the application running.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

The `requirements.txt` file contains all necessary Python packages.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the root directory of the project. This file will store your Gemini API key.

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual key.

### 5. ⚠️ IMPORTANT: Configure Hardcoded Paths in `pipeline.py`

The provided `pipeline.py` file contains two hardcoded paths that **you must change** for the application to work correctly.

#### a) Whisper Model Path

The original code points to a specific user's cache. You should change this to let Whisper manage the model download automatically.

-   **Open `pipeline.py`**.
-   **Find** the `transcribe_audio` function (around line 80).
-   **Locate** this line:
    ```python
    model = whisper.load_model("/Users/user/.cache/whisper/small.pt")
    ```
-   **Replace it** with one of the following options. The `"small"` model is a good balance of speed and accuracy. The first time you run the pipeline, this model will be downloaded automatically.
    ```python
    # Recommended: Let Whisper handle caching automatically
    model = whisper.load_model("small") 
    ```

#### b) One-Shot Example Transcript Path

The AI prompt relies on an example transcript to guide its analysis. The original path is specific to the developer's machine.

-   **Create a new folder** named `data` in the root of your project directory.
-   **Find a sample transcript** of a soccer match online and save it as a `.txt` file inside the `data` folder. For example, name it `case_study_transcript.txt`.
-   **Open `pipeline.py`**.
-   **Find** the `extract_events_with_llm` function (around line 105).
-   **Locate** this line:
    ```python
    case_study_path = "/Full Match： Belgium v Japan (2018 FIFA World Cup)/transcript.txt"
    ```
-   **Replace it** with the path to the file you just created.
    ```python
    # Update this path to point to your example transcript
    case_study_path = "data/case_study_transcript.txt"
    ```

---

## How to Run the Application

Once you have completed the setup, start the Streamlit web server from your terminal:

```bash
streamlit run streamlit_app.py
```

Your web browser should automatically open to the application's URL (usually `http://localhost:8501`).

---

## How to Use the App

1.  **Upload Video:** Use the file uploader to select a full soccer match video (`.mp4`, `.mov`, `.mkv`).
2.  **(Optional) Upload Transcript:** If you already have a clean transcript, check the box and upload the `.txt` file to skip the audio extraction and transcription steps.
3.  **Generate:** Click the "Generate Highlights & Stats" button.
4.  **Monitor Progress:** A progress bar will show the current stage of the pipeline. This process can take a long time, especially the transcription step, depending on the video length and your computer's hardware.
5.  **View Results:** Once complete, the interface will display:
    -   The final chronological highlight reel.
    -   Expandable sections for category-based highlights (e.g., all goals).
    -   A button to navigate to the statistical dashboard.
    -   Download buttons for all generated videos.
    - All clips and summaries are exported into a timestamped folder inside `Highlight_outputs/`.
6.  **Create Custom Reels:** Use the "Create Custom Highlight" tool on the right to select specific events and stitch them into a new video.

---

## Project File Structure

```
uploads/<task_id>/
  ├── <uploaded_video>.mp4
  ├── audio.wav
  ├── transcript.txt
  ├── events.json
  ├── statistics.json
  ├── clips/
  │    ├── prologue/clip_1_prologue.mp4
  │    ├── prologue/clip_1_prologue.jpg
  │    └── goal/clip_3_goal.mp4
  ├── summary_chronological.mp4
  ├── summary_goal.mp4
  ├── summary_custom_*.mp4
  └── status.json

Highlight_outputs/<match_label>_<timestamp>/  # exported copies for easy sharing
```

---

## Troubleshooting

-   **`ffmpeg.Error` or "ffmpeg not found"**: This means `ffmpeg` is not installed or not in your system's PATH. Refer to the **Prerequisites** section for installation instructions and ensure you can run `ffmpeg` from your terminal.
-   **`ValueError: GEMINI_API_KEY not set...`**: Your API key is missing. Make sure you have created the `.env` file in the project's root directory and that it contains your key in the correct format.
-   **`FileNotFoundError: ... case_study_transcript.txt`**: The application cannot find the example transcript for the AI prompt. Please follow **Step 5b** of the setup instructions carefully.
-   **Slow Performance**: The transcription step is computationally intensive. For faster processing (at the cost of some accuracy), you can change the Whisper model in `pipeline.py` from `"small"` to `"base"` or `"tiny"`.