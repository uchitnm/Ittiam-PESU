# Ittiam-PESU

Contents in this repo include model training notebooks, utilities, The End-end pipeline, and experiments for OCR and object/keypoint detection using YOLO models.

## Repository structure

- `Exploring_Yolo/`
	- `Football-OCR-main/` — Scripts and a notebook for experimenting with optical character recognition on football images. Key files:
	- `Soccer-AI/` — Main project for player/ball/pitch detectors, training notebooks and utilities.
		
- `Pipeline - Stable V3.3/` — A  pipeline and Streamlit app to run inference and visualize results.
	- `pipeline.py` — Pipeline orchestration logic used by the Streamlit app.
	- `streamlit_app.py` — Streamlit front-end to load an image/video and run detection/visualization.
	- `requirements.txt` — Python dependencies for the pipeline and app.
	- `uploads/` — (Git-ignored in normal flows) folder for user-uploaded files while using the app.

- `Gemini Exploration/` — Misc exploration and prompt experiments (notes and README).

- `Reports/` — Project reports, summaries and deliverables.

