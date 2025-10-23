Soccer-AI — Training and results for object detection & pose/keypoint models

Overview

This workspace contains notebooks and trained weights used to train object detectors (players, ball) for soccer/football video/image data. The repository stores training notebooks, pre-trained YOLO models, and a Results/ folder with experiment outputs and best/last weights.

Repository structure

- train_ball_detector.ipynb — Notebook used to train a ball detector.
- train_pitch_keypoint_detector.ipynb — Notebook used to train pitch keypoint detector (pose/keypoints for the field).
- train_player_detector.ipynb — Notebook used to train player detector.
- yolo11l.pt, yolo11n.pt, yolo11s.pt — Local YOLO model checkpoint files (various sizes).
- yolov8x-pose.pt — YOLOv8x pose-capable checkpoint.
- Results/ — Directory containing experiment folders. Each experiment contains an args.yaml, results.csv and weights/ with best.pt and last.pt.
    - football_training_b6_e25_s/
    - football_training_ball_b12_e25_s/
    - football_training_pitch_b16_e50_8x/

Quick start

Requirements

- Python 3.8+ 

Open the notebooks

1. Launch Jupyter or VS Code and open the notebooks: `train_player_detector.ipynb`, `train_ball_detector.ipynb`, `train_pitch_keypoint_detector.ipynb`.
2. Inspect the configuration cells near the top of each notebook to verify dataset paths, device (cpu / cuda), and hyperparameters.
3. Run the cells to train or evaluate models. Training will write outputs under `Results/<experiment>/weights/` and a `results.csv` file.

Using existing weights

- Best and last weights for each experiment are under `Results/<experiment>/weights/{best.pt,last.pt}`. You can load them into YOLO or your inference script.

Example inference (outline)

- If you use ultralytics YOLOv8 Python API:

  from ultralytics import YOLO
  model = YOLO('Results/football_training_b6_e25_s/weights/best.pt')
  results = model.predict(source='path/to/image_or_video', device=0)

