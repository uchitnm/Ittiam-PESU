# pipeline.py

import os
import json
import logging
import re
import datetime
import time

from dotenv import load_dotenv
load_dotenv()

import whisper
import ffmpeg
import google.generativeai as genai
from bs4 import BeautifulSoup
from markdown import markdown

# --- Logging Setup ---
# Configures a simple logger initially.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [GLOBAL] - %(message)s'
)

# --- Configure Gemini API ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY not set or is still the placeholder value in .env file.")
    genai.configure(api_key=api_key)
except ValueError as e:
    logging.warning(f"Gemini API not configured: {e}")
    api_key = None

# --- Helper Functions ---

def get_logger_with_task_id(task_id):
    """Creates a logger adapter to inject the task_id into log records."""
    return logging.LoggerAdapter(logging.getLogger(__name__), {'task_id': task_id})

def time_str_to_seconds(time_str):
    """Converts a 'hh:mm:ss' string to total seconds."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        h, m, s_float = map(float, time_str.split(':'))
        return int(h * 3600 + m * 60 + s_float)

def markdown_to_text(markdown_string):
    """Converts a markdown string to plaintext for reliable parsing."""
    html = markdown(markdown_string)
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(string=True))
    return text

# --- Pipeline Stages ---

def extract_audio(video_path, task_id):
    logger = get_logger_with_task_id(task_id)
    logger.info("Starting audio extraction...")
    task_dir = os.path.dirname(video_path)
    audio_path = os.path.join(task_dir, "audio.wav")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ar='16000', ac=1)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        logger.info("Audio extracted successfully.")
        return audio_path
    except ffmpeg.Error as e:
        logger.error(f"Audio extraction failed: {e.stderr.decode()}")
        return None

def transcribe_audio(audio_path, task_id):
    logger = get_logger_with_task_id(task_id)
    logger.info("Starting transcription with local 'large' Whisper model...")
    try:
        model = whisper.load_model("/Users/user/.cache/whisper/small.pt")
        print("Model Loaded")
        result = model.transcribe(audio_path, fp16=False)
        print("Model Used")
        task_dir = os.path.dirname(audio_path)
        transcript_path = os.path.join(task_dir, "transcript.txt")
        print("Transcript Started")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                start_time = str(datetime.timedelta(seconds=int(segment['start'])))
                text = segment['text']
                f.write(f"[{start_time}] {text.strip()}\n")
        logger.info("Transcription complete.")
        return transcript_path
    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        return None

def extract_events_with_llm(transcript_path, task_id):
    logger = get_logger_with_task_id(task_id)
    logger.info("Starting event extraction with Gemini (using one-shot highlight prompt)...")
    if not api_key:
        logger.error("Cannot proceed: GEMINI_API_KEY not configured.")
        return None
    try:
        # 1. Read the new transcript that needs to be analyzed
        with open(transcript_path, 'r', encoding='utf-8') as f:
            new_transcript_to_analyze = f.read()

        # 2. Define the path to your case study transcript
        # IMPORTANT: This path must be correct. Make it relative or absolute as needed for your setup.
        # For example, if it's in a 'data' folder next to your script:
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # case_study_path = os.path.join(script_dir, "data", "belgium_vs_japan_transcript.txt")
        # For now, I'll use a placeholder you must replace.
        case_study_path = "/Users/uchitnm/Documents/Capstone/Dataset/Full Match： Belgium v Japan (2018 FIFA World Cup)/tactiq-free-transcript-GrkiZjoyugA.txt"
        
        # Check if the case study file exists before proceeding
        if not os.path.exists(case_study_path):
            logger.error(f"FATAL: One-shot case study transcript not found at {case_study_path}")
            raise FileNotFoundError(f"Case study file not found: {case_study_path}")

        with open(case_study_path, 'r', encoding='utf-8') as f:
            case_study_transcript = f.read()

        # 3. Construct the final one-shot prompt
        prompt = f"""
You are an elite AI sports video editor, a master of narrative and pacing, who generates highlights strictly of 780 seconds, no more no less, exactly 13 minutes. Your mission is to analyze a raw sports match transcript and generate a preliminary Edit Decision List (EDL) for a compelling, world-class highlight reel.



Your guiding philosophy is to treat each extracted event as a self-contained mini-story. Every clip must have a logical beginning (the build-up), a clear climax (the event itself), and a complete resolution (the immediate aftermath, all broadcast replays, and expert commentary).

You must follow these instructions with absolute precision.

### I. CORE INSTRUCTIONS

1.  **Events to Extract**: Identify and timestamp Prologue, Goal, Foul, Missed Goal, and Epilogue. 
2.  **Clip Boundaries**: Start clips at the beginning of the meaningful build-up. End clips only after all replays and commentary analysis are complete.
3.  **Output Format**: Your entire response must be a list of EDL entries. Each entry must be on a new line and strictly adhere to this format:
    `[start hh:mm:ss] - [end hh:mm:ss] - [Team Name] - [Type] - [Short, insightful description]`

---
### II. CASE STUDY (ONE-SHOT EXAMPLE)


Here is an example of a perfect analysis. Study it carefully.

**SOURCE TRANSCRIPT FOR CASE STUDY:**

{case_study_transcript}

**PERFECT OUTPUT FOR CASE STUDY:**

[00:00:00] - [00:01:34] - N/A - prologue - Belgium makes 10 changes; Kompany gets first start.
[00:30:22] - [00:30:41] - Belgium - missed goal - Hazard's fierce shot is saved by Kawashima.
[00:42:54] - [00:43:24] - Japan - foul - Shibasaki is booked for a foul on Hazard.
[00:47:27] - [00:47:58] - Belgium - missed goal - Courtois fumbles but recovers a cross under pressure.
[00:49:58] - [00:50:36] - N/A - half-time - The first half ends scoreless.
[00:52:42] - [00:53:40] - Japan - goal - Haraguchi capitalizes on a defensive error and scores.
[00:54:15] - [00:54:30] - Belgium - missed goal - Hazard's powerful shot smashes against the post.
[00:57:00] - [00:58:18] - Japan - goal - Inui scores with a spectacular long-range strike.
[01:07:08] - [01:07:31] - Belgium - missed goal - Lukaku misses a clear header from close range.
[01:09:22] - [01:09:43] - Japan - missed goal - Courtois saves a shot from Sakai with his leg.
[01:09:43] - [01:10:18] - Belgium - replacement - Fellaini and Chadli come on for Mertens and Carrasco.
[01:14:13] - [01:15:15] - Belgium - goal - Vertonghen scores with an incredible looping header.
[01:18:45] - [01:19:48] - Belgium - goal - Substitute Fellaini heads in Hazard's cross to equalize.
[01:25:40] - [01:26:21] - Japan - replacement - Honda and Yamaguchi on for Haraguchi and Shibasaki.
[01:30:28] - [01:30:54] - Belgium - missed goal - Kawashima makes a brilliant double save on Chadli and Lukaku.
[01:38:22] - [01:38:43] - Japan - missed goal - Honda's long-range free-kick is saved by Courtois.
[01:38:58] - [01:40:33] - Belgium - goal - Courtois starts a counter-attack, finished by Chadli to win it.
[01:40:33] - [01:41:53] - N/A - epilogue - Belgium wins 3-2 after a dramatic comeback; final scenes.

---


### III. YOUR TASK

Now, using the exact same principles and format demonstrated in the case study, analyze the following new transcript and provide the EDL.

**NEW TRANSCRIPT TO ANALYZE:**

{new_transcript_to_analyze}


**YOUR OUTPUT:**
"""
        


        # Using a model that handles long context and complex instructions well.
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        logger.debug(f"Raw LLM Response:\n{response.text}")
        
        raw_text = markdown_to_text(response.text)
        timestamp_lines = [line.strip() for line in raw_text.split('\n') if line.strip() and '-' in line]
        
        if not timestamp_lines:
             logger.warning("LLM response contained no lines with timestamps. The prompt might need refinement or the model failed to find events.")
             return None

        events = []
        # A more robust regex that handles optional brackets and variations in spacing
        pattern = r'\[?(\d{1,2}:\d{2}:\d{2})\]?\s*[-–]\s*\[?(\d{1,2}:\d{2}:\d{2})\]?\s*[-–]\s*([^-]+?)\s*[-–]\s*([^-]+?)\s*[-–]\s*(.+)'
        for line in timestamp_lines:
            match = re.match(pattern, line)
            if match:
                start, end, team, event_type, desc = [g.strip() for g in match.groups()]
                # Adding a simple check to filter out obviously wrong lines
                if event_type.lower() in ["goal", "foul", "missed goal", "prologue", "epilogue", "replacement"]:
                     events.append({"start_timestamp": start, "end_timestamp": end, "team_name": team, "event_type": event_type.capitalize(), "description": desc})
        
        if not events:
            logger.warning("LLM did not return any parsable events for highlights. Check the raw response in the logs.")
            return None
            
        events.sort(key=lambda x: time_str_to_seconds(x["start_timestamp"]))
        
        # (Optional but Recommended) Duration Check
        total_duration = sum(time_str_to_seconds(e['end_timestamp']) - time_str_to_seconds(e['start_timestamp']) for e in events)
        logger.info(f"Initial extracted duration: {total_duration} seconds.")
        # Here you would implement the duration adjustment logic if needed.

        task_dir = os.path.dirname(transcript_path)
        events_path = os.path.join(task_dir, "events.json")
        with open(events_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2)
        logger.info(f"Successfully extracted and parsed {len(events)} events.")
        return events_path
        
    except Exception as e:
        logger.error(f"An error occurred during LLM event extraction: {e}", exc_info=True)
        return None
    


def generate_statistics_with_llm(transcript_path, task_id):
    """Makes a new LLM call to extract detailed match statistics."""
    logger = get_logger_with_task_id(task_id)
    logger.info("Starting statistics generation with Gemini AI...")
    if not api_key:
        logger.error("Cannot proceed with statistics: GEMINI_API_KEY not configured.")
        return None
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        prompt = f"""
Context:
---
{transcript}
---
Question:
Analyze the entire match transcript and provide a detailed statistical summary. I need you to act as an expert sports analyst. Extract the following information and return it ONLY as a single, valid JSON object. Do not include any text or markdown before or after the JSON object.
The JSON object must have the following structure:
{{
  "match_summary": "A brief, 2-3 sentence narrative summary of the match flow and result.",
  "team_stats": {{
    "team_A": {{"team_name": "Name of Team A", "score": 0, "corners": 0, "fouls": 0, "yellow_cards": 0, "red_cards": 0}},
    "team_B": {{"team_name": "Name of Team B", "score": 0, "corners": 0, "fouls": 0, "yellow_cards": 0, "red_cards": 0}}
  }},
  "player_events": {{
    "scorers": [],
    "cards_issued": []
  }}
}}
Analyze the text carefully to provide the most accurate numbers possible. If a stat is not mentioned, return 0 or an empty list.
"""
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        stats_data = json.loads(json_text)
        task_dir = os.path.dirname(transcript_path)
        stats_path = os.path.join(task_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        logger.info("Successfully generated and saved match statistics.")
        return stats_path
    except Exception as e:
        logger.error(f"An error occurred during statistics generation: {e}")
        return None

# In pipeline.py

# ... (keep all other functions as they are) ...

def create_clips_from_events(events_path, video_path, task_id):
    """
    Creates individual clips with text overlays and extracts a thumbnail for each clip.
    Specifically handles the 'prologue' to be exactly 1 minute long.
    """
    logger = get_logger_with_task_id(task_id)
    logger.info("Starting clip creation with text overlays...")
    try:
        with open(events_path, 'r') as f:
            events = json.load(f)
        
        task_dir = os.path.dirname(video_path)
        created_clips_info = []

        for i, event in enumerate(events):
            event_type_lower = event['event_type'].lower().replace(' ', '_')
            output_dir = os.path.join(task_dir, 'clips', event_type_lower)
            os.makedirs(output_dir, exist_ok=True)
            clip_filename = f"clip_{i+1}_{event_type_lower}.mp4"
            output_path = os.path.join(output_dir, clip_filename)

            overlay_text = f"{event['event_type'].upper()}: {event['team_name']} - {event['description']}"
            
            # --- MODIFICATION FOR PROLOGUE RULE ---
            start_s = time_str_to_seconds(event['start_timestamp'])
            end_s = time_str_to_seconds(event['end_timestamp'])

            if event['event_type'].lower() == 'prologue':
                logger.info(f"Applying special 1-minute rule for prologue clip.")
                # The new start time is 60 seconds before the original end time
                start_s = max(0, end_s - 60) # Use max(0, ...) to prevent negative start time
                duration = end_s - start_s
            else:
                # Original logic for all other event types
                duration = end_s - start_s
            # ----------------------------------------
            
            if duration <= 0:
                logger.warning(f"Skipping event with invalid or zero duration: {event}")
                continue

            logger.info(f"Creating clip {i+1}/{len(events)}: {clip_filename} (Duration: {duration}s)")

            input_stream = ffmpeg.input(video_path, ss=start_s, t=duration)
            video = input_stream.video.drawtext(
                text=overlay_text,
                fontsize=48,
                fontcolor='white',
                box=1,
                boxcolor='black@0.5',
                x='(w-text_w)/2',
                y='h-th-20',
                # Show the text for the entire duration of the clip, starting from the first second.
                enable=f'between(t,1,{max(1, duration-1)})' 
            )
            audio = input_stream.audio
            
            (
                ffmpeg.output(video, audio, output_path, vcodec='libx264', acodec='aac', audio_bitrate='192k', preset='fast', **{'map_metadata': -1, 'map': '0:a?'})
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            created_clips_info.append({"path": output_path, "type": event['event_type']})

            # --- THUMBNAIL GENERATION LOGIC ---
            try:
                thumbnail_path = output_path.replace('.mp4', '.jpg')
                (
                    ffmpeg
                    .input(output_path, ss=0.1)
                    .filter('scale', 320, -1)
                    .output(thumbnail_path, vframes=1)
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                logger.info(f"Generated thumbnail for clip {i+1}")
            except ffmpeg.Error as e:
                logger.warning(f"Could not generate thumbnail for clip {i+1}: {e.stderr.decode()}")
            # ------------------------------------

        logger.info(f"Successfully created {len(created_clips_info)} clips.")
        return created_clips_info
    except Exception as e:
        logger.error(f"An error occurred during clip creation: {e}", exc_info=True)
        if isinstance(e, ffmpeg.Error):
             logger.error(f"FFMPEG stderr: {e.stderr.decode()}")
        return None
    """
    Creates individual clips with text overlays and extracts a thumbnail for each clip.
    """
    logger = get_logger_with_task_id(task_id)
    logger.info("Starting clip creation with text overlays...")
    try:
        with open(events_path, 'r') as f:
            events = json.load(f)
        
        task_dir = os.path.dirname(video_path)
        created_clips_info = []

        for i, event in enumerate(events):
            event_type_lower = event['event_type'].lower().replace(' ', '_')
            output_dir = os.path.join(task_dir, 'clips', event_type_lower)
            os.makedirs(output_dir, exist_ok=True)
            clip_filename = f"clip_{i+1}_{event_type_lower}.mp4"
            output_path = os.path.join(output_dir, clip_filename)

            overlay_text = f"{event['event_type'].upper()}: {event['team_name']} - {event['description']}"
            start_s = time_str_to_seconds(event['start_timestamp'])
            end_s = time_str_to_seconds(event['end_timestamp'])
            duration = end_s - start_s
            
            if duration <= 0:
                logger.warning(f"Skipping event with invalid duration: {event}")
                continue

            logger.info(f"Creating clip {i+1}/{len(events)}: {clip_filename}")

            input_stream = ffmpeg.input(video_path, ss=start_s, t=duration)
            video = input_stream.video.drawtext(
                text=overlay_text, fontsize=48, fontcolor='white', box=1, boxcolor='black@0.5',
                x='(w-text_w)/2', y='h-th-20', enable=f'between(t,1,{max(1, duration-1)})'
            )
            audio = input_stream.audio
            
            (
                ffmpeg.output(video, audio, output_path, vcodec='libx264', acodec='aac', audio_bitrate='192k', preset='fast', **{'map_metadata': -1, 'map': '0:a?'})
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            created_clips_info.append({"path": output_path, "type": event['event_type']})

            # --- THUMBNAIL GENERATION LOGIC ---
            try:
                thumbnail_path = output_path.replace('.mp4', '.jpg')
                (
                    ffmpeg
                    .input(output_path, ss=0.1) # Take frame from 0.1s to avoid black start frames
                    .filter('scale', 320, -1)  # Scale to a reasonable width, maintain aspect ratio
                    .output(thumbnail_path, vframes=1) # Output a single frame
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                logger.info(f"Generated thumbnail for clip {i+1}")
            except ffmpeg.Error as e:
                logger.warning(f"Could not generate thumbnail for clip {i+1}: {e.stderr.decode()}")
            # ------------------------------------

        logger.info(f"Successfully created {len(created_clips_info)} clips.")
        return created_clips_info
    except Exception as e:
        logger.error(f"An error occurred during clip creation: {e}")
        if isinstance(e, ffmpeg.Error):
             logger.error(f"FFMPEG stderr: {e.stderr.decode()}")
        return None

def stitch_clips(clip_paths, output_path, task_id):
    logger = get_logger_with_task_id(task_id)
    if not clip_paths:
        logger.warning("No clips to stitch.")
        return None
    logger.info(f"Starting to stitch {len(clip_paths)} clips into {os.path.basename(output_path)}.")
    try:
        processed_streams = []
        for path in clip_paths:
            probe = ffmpeg.probe(path)
            has_audio = any(s.get('codec_type') == 'audio' for s in probe['streams'])
            video_stream = ffmpeg.input(path).video
            audio_stream = None
            if has_audio:
                audio_stream = ffmpeg.input(path).audio
            else:
                logger.warning(f"Clip {os.path.basename(path)} has no audio. Generating silent track.")
                duration = float(probe['format']['duration'])
                audio_stream = ffmpeg.input(f'anullsrc=r=44100:cl=mono:d={duration}', f='lavfi').audio
            processed_streams.append(video_stream)
            processed_streams.append(audio_stream)
        concatenated = ffmpeg.concat(*processed_streams, v=1, a=1).node
        video_out, audio_out = concatenated[0], concatenated[1]
        (ffmpeg.output(video_out, audio_out, output_path, vcodec='libx264', acodec='aac', preset='fast').run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
        logger.info(f"Stitching complete for {os.path.basename(output_path)}.")
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"Stitching failed for {os.path.basename(output_path)}: {e.stderr.decode()}")
        return None

def run_full_pipeline(video_path, task_id, status_callback, start_time=None):
    handler = logging.getLogger().handlers[0]
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(task_id)s] - %(message)s'))
    logger = get_logger_with_task_id(task_id)
    task_dir = os.path.dirname(video_path)
    try:
        # 1. Extract audio (skip if transcript.txt exists)
        transcript_path = os.path.join(task_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            logger.info("Transcript already exists. Skipping audio extraction and transcription.")
        else:
            audio_path = os.path.join(task_dir, "audio.wav")
            if os.path.exists(audio_path):
                logger.info("Audio already extracted. Skipping.")
            else:
                status_callback("Extracting audio...")
                audio_path = extract_audio(video_path, task_id)
                if not audio_path: raise Exception("Audio extraction failed")
            # 2. Transcribe audio
            status_callback("Transcribing audio... (this may take a while)")
            transcript_path = transcribe_audio(audio_path, task_id)
            if not transcript_path: raise Exception("Transcription failed")

        # 3. Extract events
        events_path = os.path.join(task_dir, "events.json")
        if os.path.exists(events_path):
            logger.info("Events already extracted. Skipping event extraction.")
        else:
            status_callback("Identifying key events with AI...")
            events_path = extract_events_with_llm(transcript_path, task_id)
            if not events_path: raise Exception("AI event extraction failed")

        # 4. Generate statistics
        stats_path = os.path.join(task_dir, "statistics.json")
        if os.path.exists(stats_path):
            logger.info("Statistics already generated. Skipping statistics generation.")
        else:
            status_callback("Generating match statistics...")
            generate_statistics_with_llm(transcript_path, task_id)

        # 5. Create clips
        clips_dir = os.path.join(task_dir, 'clips')
        # Check if all clips exist by comparing with events
        with open(events_path, 'r') as f:
            events = json.load(f)
        all_clips_exist = True
        clips_info = []
        for i, event in enumerate(events):
            event_type_lower = event['event_type'].lower().replace(' ', '_')
            output_dir = os.path.join(clips_dir, event_type_lower)
            clip_filename = f"clip_{i+1}_{event_type_lower}.mp4"
            output_path = os.path.join(output_dir, clip_filename)
            if not os.path.exists(output_path):
                all_clips_exist = False
            clips_info.append({"path": output_path, "type": event['event_type']})
        if all_clips_exist:
            logger.info("All clips already exist. Skipping clip creation.")
        else:
            status_callback("Creating individual highlight clips with overlays...")
            clips_info = create_clips_from_events(events_path, video_path, task_id)
            if not clips_info: raise Exception("Clip creation failed")

        # 6. Stitch main chronological summary
        summary_path = os.path.join(task_dir, "summary_chronological.mp4")
        if os.path.exists(summary_path):
            logger.info("Main summary already exists. Skipping stitching.")
        else:
            status_callback("Stitching main chronological summary...")
            all_clip_paths = [c['path'] for c in clips_info]
            stitch_clips(all_clip_paths, summary_path, task_id)

        # 7. Stitch category-based summaries
        # Check for each category summary
        status_callback("Stitching category-based summaries...")
        clips_by_category = {}
        for clip in clips_info:
            cat = clip['type']
            if cat not in clips_by_category:
                clips_by_category[cat] = []
            clips_by_category[cat].append(clip['path'])
        for category, clips in clips_by_category.items():
            cat_filename = f"summary_{category.lower().replace(' ', '_')}.mp4"
            cat_output_path = os.path.join(task_dir, cat_filename)
            if os.path.exists(cat_output_path):
                logger.info(f"Category summary {cat_filename} already exists. Skipping.")
            else:
                stitch_clips(clips, cat_output_path, task_id)

        status_callback("Complete", final_summary_path=summary_path, elapsed_time=(time.time() - start_time) if start_time else None)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        status_callback(f"Error: {e}")
    finally:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [GLOBAL] - %(message)s'))