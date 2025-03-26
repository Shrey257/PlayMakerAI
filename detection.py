import cv2
import numpy as np
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", device=None):
        """
        Initialize the object detector with YOLOv8
        
        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        
        # Classes for filtering
        self.player_class_ids = [0]  # person class in COCO dataset
        self.ball_class_ids = [32]   # sports ball class in COCO dataset
        
    def detect(self, frame, conf_threshold=0.25):
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            players: List of player detections [x1, y1, x2, y2, conf, class_id]
            balls: List of ball detections [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        players = []
        balls = []
        
        # Extract detections
        for detection in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)
            
            # Filter by class
            if class_id in self.player_class_ids:
                players.append(detection)
            elif class_id in self.ball_class_ids:
                balls.append(detection)
                
        return players, balls
    
    def train_custom_model(self, data_yaml, epochs=50, img_size=640, batch_size=16):
        """
        Train a custom YOLOv8 model on football dataset
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size for training
        """
        # Create a new YOLOv8 model instance for training
        model = YOLO('yolov8n.pt')
        
        # Train the model
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='football_detector'
        )
        
        return model


class ObjectTracker:
    def __init__(self, tracker_type="bytetrack"):
        """
        Initialize object tracker
        
        Args:
            tracker_type: Type of tracker to use
        """
        self.tracker_type = tracker_type
        
    def track(self, detections, frame):
        """
        Track objects across frames
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf, class_id]
            frame: Current frame
            
        Returns:
            tracks: List of tracks with track_id
        """
        # This would be implemented with a proper tracking library
        # For simplicity, we return the detections as is
        return detections 