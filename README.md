# Football Analysis System

A computer vision system for football analysis using YOLOv8, object tracking, team assignment, and player metrics calculation.

## Features

- **Object Detection:** Uses YOLOv8 to detect players, referees, and the ball.
- **Custom Training:** Train your own object detector with labeled datasets.
- **Team Assignment:** Classifies players into teams based on jersey colors using KMeans clustering.
- **Camera Tracking:** Detects camera movement using optical flow.
- **Perspective Transformation:** Ensures accurate distance measurements.
- **Player Metrics:** Calculates player speed and distance traveled.

## Installation

### Prerequisites
Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Steps

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/football-analysis-system.git
   cd football-analysis-system
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Analysis
Run the main script with a video input:
```sh
python main.py --video path/to/video.mp4 --show
```
- `--video`: Path to the input video.
- `--show`: (Optional) Display the output during processing.

### Training a Custom Model
1. **Prepare the dataset:**
   ```sh
   python prepare_dataset.py --roboflow-dir path/to/roboflow/dataset --output-dir data
   ```
2. **Train the YOLOv8 detector:**
   ```sh
   python train_detector.py --data data/data.yaml
   ```

## Project Structure
```
football-analysis-system/
├── data/                  # Dataset and annotations
├── models/                # Trained models and weights
├── src/                   # Source code for analysis
│   ├── tracking.py        # Object tracking module
│   ├── detection.py       # YOLOv8 detection
│   ├── metrics.py         # Speed and distance calculations
│   ├── perspective.py     # Perspective transformation
│   ├── team_assignment.py # Team color classification
├── main.py                # Main analysis script
├── prepare_dataset.py     # Dataset preprocessing
├── train_detector.py      # Model training script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For issues or feature requests, open an issue on GitHub or contact me at **your.email@example.com**.

