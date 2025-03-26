# Football Analysis System

A computer vision system for football analysis using YOLOv8, object tracking, team assignment, and player metrics calculation.

## Features

- Object detection using YOLOv8 (players, referees, balls)
- Custom object detector training
- Team assignment based on jersey colors using KMeans clustering
- Camera movement tracking with optical flow
- Perspective transformation for accurate distance measurements
- Player speed and distance calculations

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/football-analysis-system.git
   cd football-analysis-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main analysis script:
   ```
   python main.py --video path/to/video.mp4 --show
   ```

2. For custom model training:
   ```
   python prepare_dataset.py --roboflow-dir path/to/roboflow/dataset --output-dir data
   python train_detector.py --data data/data.yaml
   ```#   P l a y M a k e r - A I  
 