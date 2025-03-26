import cv2
import numpy as np
import argparse
import time
from collections import defaultdict

from detection import ObjectDetector, ObjectTracker
from team_assignment import TeamAssigner
from camera_tracking import CameraTracker
from perspective import PerspectiveTransformer
from player_metrics import PlayerMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Football Analysis System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='Path to output video file')
    parser.add_argument('--show', action='store_true', help='Show video output')
    parser.add_argument('--field-width', type=float, default=105, help='Football field width in meters')
    parser.add_argument('--field-height', type=float, default=68, help='Football field height in meters')
    return parser.parse_args()

def draw_results(frame, players, balls, team_labels, player_metrics, track_ids, team_colors):
    """Draw detection and tracking results on the frame"""
    # Draw players
    for i, (player, team_id, track_id) in enumerate(zip(players, team_labels, track_ids)):
        x1, y1, x2, y2 = map(int, player[:4])
        
        # Get player speed and distance
        speed_ms, speed_kmh = player_metrics.get_player_speed(track_id)
        distance = player_metrics.get_player_distance(track_id)
        
        # Draw bounding box with team color
        color = (0, 0, 255)  # Default color (red)
        if team_id == 0:
            color = (0, 0, 255)  # Team 1 (red)
        elif team_id == 1:
            color = (255, 0, 0)  # Team 2 (blue)
        elif team_id == 2:
            color = (0, 0, 0)    # Referee (black)
            
        if team_colors is not None and len(team_colors) > team_id:
            color = tuple(map(int, team_colors[team_id]))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw player ID and speed
        speed_text = f"{track_id}: {speed_kmh:.1f} km/h"
        cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Draw balls
    for ball in balls:
        x1, y1, x2, y2 = map(int, ball[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
    return frame

def main():
    args = parse_args()
    
    # Initialize components
    detector = ObjectDetector(model_path=args.model)
    tracker = ObjectTracker()
    team_assigner = TeamAssigner()
    camera_tracker = CameraTracker()
    perspective_transformer = PerspectiveTransformer(
        field_width=args.field_width, 
        field_height=args.field_height
    )
    
    # Open video file
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize player metrics calculator
    player_metrics = PlayerMetrics(fps=fps)
    
    # Initialize video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Track IDs for players
    next_track_id = 0
    player_track_ids = {}
    
    # Process video frames
    frame_count = 0
    perspective_set = False
    team_colors = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects
        players, balls = detector.detect(frame, conf_threshold=args.conf)
        
        # Track camera movement
        camera_movement, homography = camera_tracker.estimate_camera_movement(frame)
        
        # Initialize perspective transform on first frame with good detections
        if not perspective_set and len(players) > 0:
            field_points = perspective_transformer.estimate_field_points_from_lines(frame)
            perspective_transformer.set_field_points(field_points)
            perspective_set = True
        
        # Assign team labels
        if len(players) > 0:
            team_labels, valid_players = team_assigner.assign_teams(frame, players)
            players = valid_players
            team_colors = team_assigner.team_colors
        else:
            team_labels = []
        
        # Assign track IDs for players
        track_ids = []
        for i, player in enumerate(players):
            x1, y1, x2, y2 = map(int, player[:4])
            player_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Simple tracking based on IoU - a real implementation would use a more
            # sophisticated tracker like ByteTrack, DeepSORT, etc.
            best_iou = 0
            best_id = None
            
            for track_id, (prev_x1, prev_y1, prev_x2, prev_y2) in player_track_ids.items():
                # Calculate IoU
                x_left = max(x1, prev_x1)
                y_top = max(y1, prev_y1)
                x_right = min(x2, prev_x2)
                y_bottom = min(y2, prev_y2)
                
                if x_right < x_left or y_bottom < y_top:
                    continue
                    
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
                iou = intersection / (area1 + area2 - intersection)
                
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is None:
                best_id = next_track_id
                next_track_id += 1
                
            track_ids.append(best_id)
            player_track_ids[best_id] = (x1, y1, x2, y2)
            
            # Transform player position to field coordinates
            if perspective_set:
                player_position_pixels = (player_center[0], player_center[1])
                player_position_meters = perspective_transformer.transform_point(player_position_pixels)
                
                # Update player metrics
                player_metrics.update_player_position(best_id, player_position_meters)
        
        # Draw results on frame
        result_frame = draw_results(
            frame.copy(), players, balls, team_labels, 
            player_metrics, track_ids, team_colors
        )
        
        # Add visualization of the perspective transformation
        if perspective_set:
            perspective_viz = perspective_transformer.visualize_transformation(frame)
            perspective_viz = cv2.resize(perspective_viz, (width, height//3))
            result_frame[height-perspective_viz.shape[0]:, :] = perspective_viz
        
        # Display frame info
        cv2.putText(
            result_frame, 
            f"Frame: {frame_count} | Players: {len(players)} | Balls: {len(balls)}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Write or display output
        if writer:
            writer.write(result_frame)
            
        if args.show:
            cv2.imshow('Football Analysis', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_count += 1
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 