# streamlit_app.py

import streamlit as st
import os
import time
import json
import threading
from glob import glob
import pandas as pd
import datetime
import shutil

import pipeline
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Soccer Highlight Pro",
    page_icon="⚽",
    layout="wide"
)

# --- App State Management ---
if "view" not in st.session_state: st.session_state.view = 'generator'
if "current_task_id" not in st.session_state: st.session_state.current_task_id = None
if "processing" not in st.session_state: st.session_state.processing = False
if "custom_video_path" not in st.session_state: st.session_state.custom_video_path = None

# --- Constants & Backend Helpers ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FINAL_HIGHLIGHT_FOLDER = "Highlight_outputs"
os.makedirs(FINAL_HIGHLIGHT_FOLDER, exist_ok=True)



def get_task_dir(task_id): return os.path.join(UPLOAD_FOLDER, task_id)
def get_status(task_id):
    status_file = os.path.join(get_task_dir(task_id), 'status.json')
    if not os.path.exists(status_file): return None
    try:
        with open(status_file, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return None
def update_status(task_id, status_message, final_summary_path=None, elapsed_time=None):
    task_dir, status_data = get_task_dir(task_id), {"status": status_message}
    os.makedirs(task_dir, exist_ok=True)
    if final_summary_path: status_data["final_summary_filename"] = os.path.basename(final_summary_path)
    if elapsed_time is not None:
        status_data["completion_time"] = elapsed_time
    with open(os.path.join(task_dir, 'status.json'), 'w') as f: json.dump(status_data, f)
def pipeline_thread_target(video_path, task_id):
    def status_callback(status_message, final_summary_path=None, elapsed_time=None):
        update_status(task_id, status_message, final_summary_path, elapsed_time)
    start_time = time.time()
    pipeline.run_full_pipeline(video_path, task_id, status_callback, start_time)
    st.session_state.processing = False


# --- UI Component Functions ---

def display_processing_view(current_step_text, task_id, use_transcript=False, start_time=None):
    if use_transcript:
        PIPELINE_STEPS = [
            "Identifying key events with AI...",
            "Generating match statistics...",
            "Creating individual highlight clips with overlays...",
            "Stitching main chronological summary...",
            "Stitching category-based summaries...",
            "Complete"
        ]
    else:
        PIPELINE_STEPS = [
            "Extracting audio...",
            "Transcribing audio... (this may take a while)",
            "Identifying key events with AI...",
            "Generating match statistics...",
            "Creating individual highlight clips with overlays...",
            "Stitching main chronological summary...",
            "Stitching category-based summaries...",
            "Complete"
        ]
    try: current_index = PIPELINE_STEPS.index(current_step_text)
    except ValueError: current_index = -1
    total_steps = len(PIPELINE_STEPS) - 1
    progress_percent = (current_index + 1) / total_steps if current_index >= 0 else 0
    st.markdown("""<style>.progress-container{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;}.progress-label{font-size:20px;font-weight:600;color:#6C47FF;margin:10px 0 10px 0;}</style>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="progress-container"><h2>Processing Your Video</h2></div>', unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: grey;'>Task ID: {task_id}</p>", unsafe_allow_html=True)
        progress_label = f"Processing Step {current_index + 1}/{total_steps}: {current_step_text}"
        st.markdown(f'<div class="progress-container"><span class="progress-label">{progress_label}</span></div>', unsafe_allow_html=True)
        st.progress(progress_percent)
        # --- Running timer ---
        if start_time is not None:
            elapsed = int(time.time() - start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            if hours > 0:
                timer_str = f"{hours} hr {minutes} min {seconds} sec"
            elif minutes > 0:
                timer_str = f"{minutes} min {seconds} sec"
            else:
                timer_str = f"{seconds} sec"
            st.info(f"Elapsed time: {timer_str}")
        st.markdown("<p style='text-align: center; color: grey;'>Please keep this browser tab open until processing is complete.</p>", unsafe_allow_html=True)

def display_interactive_event_creator(task_id):
    st.header("Create Custom Highlight")
    st.caption("Select events by event type and team to include in your highlight reel.")
    task_dir = get_task_dir(task_id)
    events_path = os.path.join(task_dir, "events.json")
    if not os.path.exists(events_path):
        st.warning("Event data not available."); return
    df = pd.read_json(events_path)
    st.info("Select by event type/team, or use advanced filtering for fine-grained control.")

    # --- Flat list of grouped checkboxes (no dropdowns) ---
    event_types = sorted(df['event_type'].unique())
    teams = sorted(df['team_name'].unique())
    group_select = {}
    for event_type in event_types:
        teams_in_type = sorted(df[df['event_type'] == event_type]['team_name'].unique())
        for team in teams_in_type:
            key = f"select_{event_type}_{team}_{task_id}"
            group_select[(event_type, team)] = st.checkbox(f"Include all {event_type} events for {team}", value=False, key=key)

    st.divider()
    advanced = st.checkbox("Show advanced event selection", value=False, key=f"adv_{task_id}")
    selected_indices = set()
    if advanced:
        # Show full event list with checkboxes
        df_adv = df.copy()
        df_adv['Select'] = True
        edited_df = st.data_editor(
            df_adv[['Select', 'event_type', 'team_name']],
            key=f"adv_editor_{task_id}",
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn("Include?", default=True), "event_type": "Event Type", "team_name": "Team"}
        )
        for i, row in edited_df.iterrows():
            if row['Select']:
                selected_indices.add(i)
    else:
        # Use group selections
        for idx, row in df.iterrows():
            if group_select.get((row['event_type'], row['team_name']), False):
                selected_indices.add(idx)
    st.divider()
    if st.button("Stitch Custom Video", disabled=(len(selected_indices) == 0), type="primary"):
        with st.spinner("Finding and stitching selected clips..."):
            selected_clip_paths = []
            for idx in selected_indices:
                event = df.iloc[idx]
                event_type_lower = event['event_type'].lower().replace(' ', '_')
                clip_filename = f"clip_{idx + 1}_{event_type_lower}.mp4"
                clip_path = os.path.join(task_dir, 'clips', event_type_lower, clip_filename)
                if os.path.exists(clip_path):
                    selected_clip_paths.append(clip_path)
                else:
                    st.warning(f"Could not find clip file: {clip_filename}")
            if selected_clip_paths:
                custom_output_path = os.path.join(task_dir, f"summary_custom_{int(time.time())}.mp4")
                pipeline.stitch_clips(selected_clip_paths, custom_output_path, task_id)
                st.session_state.custom_video_path = custom_output_path
                st.rerun()
            else:
                st.error("No valid clip files found for the selected events.")

def display_dashboard(stats):
    st.header(f"📊 Statistical Analysis: {stats['team_stats']['team_A']['team_name']} vs {stats['team_stats']['team_B']['team_name']}")
    # Only show Goals and Head-to-Head Stats
    col1, col2 = st.columns(2)
    team_a, team_b = stats['team_stats']['team_A'], stats['team_stats']['team_B']
    # Try to get country flag emoji from team name (simple mapping for common countries)
    def get_flag(team_name):
        flags = {
            'Brazil': '🇧🇷',
            'Germany': '🇩🇪',
            'Japan': '🇯🇵',
            'Belgium': '🇧🇪',
            'France': '🇫🇷',
            'Argentina': '🇦🇷',
            'England': '🇬🇧',
            'Spain': '🇪🇸',
            'Italy': '🇮🇹',
            'Netherlands': '🇳🇱',
            'Portugal': '🇵🇹',
            'USA': '🇺🇸',
            'United States': '🇺🇸',
            'Mexico': '🇲🇽',
            'Croatia': '🇭🇷',
            'Switzerland': '🇨🇭',
            'Uruguay': '🇺🇾',
            'Russia': '🇷🇺',
            'South Korea': '🇰🇷',
            'Korea Republic': '🇰🇷',
            'Australia': '🇦🇺',
            'Morocco': '🇲🇦',
            'Senegal': '🇸🇳',
            'Cameroon': '🇨🇲',
            'Saudi Arabia': '🇸🇦',
            'Poland': '🇵🇱',
            'Canada': '🇨🇦',
            'Ghana': '🇬🇭',
            'Ecuador': '🇪🇨',
            'Serbia': '🇷🇸',
            'Denmark': '🇩🇰',
            'Sweden': '🇸🇪',
            'Wales': '🏴',
            'Scotland': '🏴',
            'Iran': '🇮🇷',
            'Costa Rica': '🇨🇷',
            'Tunisia': '🇹🇳',
            'Qatar': '🇶🇦',
            'Egypt': '🇪🇬',
            'Nigeria': '🇳🇬',
            'Turkey': '🇹🇷',
            'Greece': '🇬🇷',
            'Chile': '🇨🇱',
            'Colombia': '🇨🇴',
            'Paraguay': '🇵🇾',
            'Peru': '🇵🇪',
            'Czech Republic': '🇨🇿',
            'Ukraine': '🇺🇦',
            'Romania': '🇷🇴',
            'Hungary': '🇭🇺',
            'Norway': '🇳🇴',
            'Ireland': '🇮🇪',
            'Ivory Coast': '🇨🇮',
            'Algeria': '🇩🇿',
            'South Africa': '🇿🇦',
            'New Zealand': '🇳🇿',
        }
        for k, v in flags.items():
            if k.lower() in team_name.lower():
                return v
        return ''
    with col1:
        flag = get_flag(team_a['team_name'])
        st.subheader(f"{flag} {team_a['team_name']}")
        st.metric("Score", team_a.get('score', 0))
    with col2:
        flag = get_flag(team_b['team_name'])
        st.subheader(f"{flag} {team_b['team_name']}")
        st.metric("Score", team_b.get('score', 0))
    st.divider(); st.markdown("### Head-to-Head Stats")
    chart_data = {
        'Stat': ['Corners', 'Fouls', 'Yellow Cards', 'Red Cards'],
        team_a['team_name']: [team_a.get('corners', 0), team_a.get('fouls', 0), team_a.get('yellow_cards', 0), team_a.get('red_cards', 0)],
        team_b['team_name']: [team_b.get('corners', 0), team_b.get('fouls', 0), team_b.get('yellow_cards', 0), team_b.get('red_cards', 0)]
    }
    st.bar_chart(pd.DataFrame(chart_data).set_index('Stat'))
    st.divider(); st.markdown("### Key Player Events")
    # Only show Goal Scorers and Cards Issued, remove Assists
    event_col1, event_col2 = st.columns(2)
    with event_col1:
        st.markdown("#### Goal Scorers")
        st.dataframe(pd.DataFrame(stats.get('player_events', {}).get('scorers', [])), hide_index=True, use_container_width=True)
    with event_col2:
        st.markdown("#### Cards Issued")
        st.dataframe(pd.DataFrame(stats.get('player_events', {}).get('cards_issued', [])), hide_index=True, use_container_width=True)

# ===================================================================
# =================== MAIN APPLICATION LOGIC ========================
# ===================================================================

if st.session_state.view == 'generator':
    st.title("⚽ Soccer Highlight Pro")
    uploaded_file = st.file_uploader("Upload a full match video to generate highlight and statistics of the Match.", type=["mp4", "mov", "mkv"])

    use_transcript = st.checkbox("I already have a transcript file (txt)")
    transcript_file = None
    if use_transcript:
        transcript_file = st.file_uploader("Upload transcript file", type=["txt"])

    if uploaded_file is not None and not st.session_state.processing:
        if st.button("Generate Highlights & Stats"):
            with st.spinner("Preparing file..."):
                temp_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                task_id = utils.get_file_hash(temp_path)
                st.session_state.current_task_id = task_id; st.session_state.custom_video_path = None
                task_dir = get_task_dir(task_id)
                status = get_status(task_id)
                if status and status.get('status') == 'Complete':
                    st.success("✅ Cache Hit! Loading previous results.")
                    os.remove(temp_path); time.sleep(1)
                else:
                    st.info("✅ New video detected. Starting the full pipeline.")
                    st.session_state.processing = True; os.makedirs(task_dir, exist_ok=True)
                    video_path = os.path.join(task_dir, uploaded_file.name)
                    os.replace(temp_path, video_path)
                    if use_transcript and transcript_file is not None:
                        transcript_path = os.path.join(task_dir, "transcript.txt")
                        with open(transcript_path, "wb") as f:
                            f.write(transcript_file.getbuffer())
                    st.session_state.use_transcript = use_transcript and transcript_file is not None
                    st.session_state.pipeline_start_time = time.time()
                    thread = threading.Thread(target=pipeline_thread_target, args=(video_path, task_id))
                    thread.start()
            st.rerun()

    st.divider()

    if st.session_state.current_task_id:
        task_id = st.session_state.current_task_id
        status = get_status(task_id)

        if not status:
            st.info("Waiting for process to initialize...")
            time.sleep(2); st.rerun()
        
        elif status.get('status') == 'Complete':
            task_dir = get_task_dir(task_id)
            stats_file = os.path.join(task_dir, "statistics.json")
            completion_time = status.get("completion_time")
            if completion_time is not None:
                # Format completion_time in HH:MM:SS or "X hr Y min Z sec"
                if completion_time >= 3600:
                    hours = int(completion_time // 3600)
                    minutes = int((completion_time % 3600) // 60)
                    seconds = int(completion_time % 60)
                    time_str = f"{hours} hr {minutes} min {seconds} sec"
                elif completion_time >= 60:
                    minutes = int(completion_time // 60)
                    seconds = int(completion_time % 60)
                    time_str = f"{minutes} min {seconds} sec"
                else:
                    time_str = f"{int(completion_time)} sec"
                st.success(f"Pipeline completed in {time_str}.")

            # --- Export final highlights to FINAL_HIGHLIGHT_FOLDER ---
            try:
                export_marker = os.path.join(task_dir, 'exported_to.txt')
                if not os.path.exists(export_marker):
                    # get the final summary filename if present
                    summary_filename = status.get('final_summary_filename')
                    # Determine a match label from statistics if possible
                    match_label = task_id
                    try:
                        if os.path.exists(stats_file):
                            with open(stats_file, 'r') as sf: stats_obj = json.load(sf)
                            ta = stats_obj.get('team_stats', {}).get('team_A', {}).get('team_name')
                            tb = stats_obj.get('team_stats', {}).get('team_B', {}).get('team_name')
                            if ta and tb:
                                def _clean(n):
                                    return ''.join(c for c in n if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
                                match_label = f"{_clean(ta)}_vs_{_clean(tb)}"
                    except Exception:
                        match_label = task_id

                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    final_subdir = f"{match_label}_{timestamp}"
                    final_dir = os.path.join(FINAL_HIGHLIGHT_FOLDER, final_subdir)
                    os.makedirs(final_dir, exist_ok=True)

                    # Collect summary videos and any generated clips
                    files_to_copy = []
                    if summary_filename:
                        sfpath = os.path.join(task_dir, summary_filename)
                        if os.path.exists(sfpath): files_to_copy.append(sfpath)
                    # category summaries
                    files_to_copy += glob(os.path.join(task_dir, "summary_*.mp4"))
                    files_to_copy += glob(os.path.join(task_dir, "summary_custom*.mp4"))
                    # individual clips in subfolders
                    files_to_copy += glob(os.path.join(task_dir, "clips", "**", "*.mp4"), recursive=True)

                    files_to_copy = sorted({p for p in files_to_copy if os.path.exists(p)})
                    for src in files_to_copy:
                        try:
                            shutil.copy2(src, os.path.join(final_dir, os.path.basename(src)))
                        except Exception as e:
                            # Non-fatal: show a warning in the UI
                            st.warning(f"Failed to copy {os.path.basename(src)}: {e}")

                    # Write marker so we don't copy again on reruns
                    try:
                        with open(export_marker, 'w') as m: m.write(final_dir)
                    except Exception:
                        pass
                    st.info(f"Saved highlight videos to: {final_dir}")
            except Exception as e:
                st.warning(f"Error while exporting highlights: {e}")
            
            col1, col2 = st.columns([2, 1], gap="large")

            with col1:
                if os.path.exists(stats_file):
                    if st.button("📊 View Statistics Dashboard", type="primary"):
                        st.session_state.view = 'dashboard'; st.rerun()
                
                st.header("Chronological Highlights")
                summary_filename = status.get('final_summary_filename')
                if summary_filename and os.path.exists(os.path.join(task_dir, summary_filename)):
                    st.video(os.path.join(task_dir, summary_filename))
                    with open(os.path.join(task_dir, summary_filename), "rb") as file: st.download_button(label=f"Download Chronological Reel", data=file, file_name=summary_filename, mime="video/mp4")
                
                if st.session_state.custom_video_path and os.path.exists(st.session_state.custom_video_path):
                    st.header("Your Custom Highlight Reel")
                    st.video(st.session_state.custom_video_path)
                    with open(st.session_state.custom_video_path, "rb") as file: st.download_button(label=f"Download Custom Reel", data=file, file_name=os.path.basename(st.session_state.custom_video_path), mime="video/mp4")

                # --- RESTORED CATEGORY HIGHLIGHTS SECTION ---
                st.header("Event-Based Highlights")
                category_summaries = sorted(glob(os.path.join(task_dir, "summary_*.mp4")))
                for vid_path in category_summaries:
                    basename = os.path.basename(vid_path)
                    # Exclude the main chronological and any custom videos from this list
                    if basename != summary_filename and not basename.startswith("summary_custom"):
                        with st.expander(f"🎬 View: {basename}"):
                            st.video(vid_path)
                            with open(vid_path, "rb") as file:
                                st.download_button(
                                    label=f"Download",
                                    data=file,
                                    file_name=basename,
                                    mime="video/mp4",
                                    key=f"dl_{basename}" # Unique key is crucial
                                )
                # ----------------------------------------------

            with col2:
                display_interactive_event_creator(task_id)
        
        elif 'Error' in status.get('status', ''):
            st.error(f"An error occurred: {status['status']}")
            st.session_state.processing = False
        
        else: # Still processing
            status_message = status.get('status', 'Initializing...')
            display_processing_view(
                status_message,
                task_id,
                use_transcript=st.session_state.get('use_transcript', False),
                start_time=st.session_state.get('pipeline_start_time')
            )
            time.sleep(3); st.rerun()

elif st.session_state.view == 'dashboard':
    st.balloons()
    if st.button("⬅️ Back to Highlights"): st.session_state.view = 'generator'; st.rerun()
    task_id = st.session_state.current_task_id
    if task_id:
        stats_file = os.path.join(get_task_dir(task_id), "statistics.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f: stats = json.load(f)
            display_dashboard(stats)
        else: st.error("Could not load statistics file.")
    else: st.warning("No task selected.")