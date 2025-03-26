import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "Raw_Data.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Process the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from video")
    exit()

# Run detection on the frame
results = model(frame)

# Print detection results
print(f"Detected {len(results[0].boxes)} objects")

# Display the frame with detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the annotated frame
cv2.imwrite("detected_frame.jpg", frame)
print("Detection test completed successfully. Check 'detected_frame.jpg' for results.")

# Clean up
cap.release()
cv2.destroyAllWindows() 