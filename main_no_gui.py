import cv2
import numpy as np
import argparse
import time
from collections import defaultdict
import os
from ultralytics import YOLO

from detection import ObjectDetector, ObjectTracker
from team_assignment import TeamAssigner
from camera_tracking import CameraTracker
from perspective import PerspectiveTransformer
from player_metrics import PlayerMetrics

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        
    def detect(self, frame, conf_threshold=0.25):
        results = self.model(frame, conf=conf_threshold)[0]
        players = []
        balls = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if class_id == 0:  # person
                players.append([x1, y1, x2, y2, score])
            elif class_id == 32:  # sports ball
                balls.append([x1, y1, x2, y2, score])
                
        return players, balls

class ObjectTracker:
    def __init__(self):
        self.prev_frame = None
        self.prev_pts = None
        
    def update(self, frame, objects):
        if len(objects) == 0:
            self.prev_frame = None
            self.prev_pts = None
            return []
            
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_pts = np.float32([[int((x1 + x2) / 2), int((y1 + y2) / 2)] for x1, y1, x2, y2, _ in objects])
        
        if self.prev_frame is None or self.prev_pts is None:
            self.prev_frame = current_frame
            self.prev_pts = current_pts
            return objects
            
        # Calculate optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, current_frame, self.prev_pts, current_pts
        )
        
        self.prev_frame = current_frame
        self.prev_pts = new_pts
        
        return objects

class TeamAssigner:
    def __init__(self):
        self.team_colors = None
        
    def assign_teams(self, frame, players):
        if len(players) == 0:
            return [], []
            
        # Extract jersey colors
        jersey_colors = []
        valid_players = []
        
        for player in players:
            x1, y1, x2, y2 = map(int, player[:4])
            if y2 - y1 < 10 or x2 - x1 < 10:  # Skip if bounding box is too small
                continue
                
            # Extract upper body region
            upper_body = frame[y1:y1 + (y2 - y1) // 3, x1:x2]
            if upper_body.size == 0:
                continue
                
            # Average color in HSV space
            hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
            avg_color = np.mean(hsv, axis=(0, 1))
            jersey_colors.append(avg_color)
            valid_players.append(player)
            
        if len(jersey_colors) == 0:
            return [], []
            
        # Cluster colors into teams
        jersey_colors = np.array(jersey_colors)
        if self.team_colors is None:
            # Initialize team colors using first frame
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(jersey_colors)
            self.team_colors = kmeans.cluster_centers_
            
        # Assign teams based on closest team color
        team_labels = []
        for color in jersey_colors:
            distances = np.linalg.norm(self.team_colors - color, axis=1)
            team_labels.append(np.argmin(distances))
            
        return team_labels, valid_players

class PlayerMetrics:
    def __init__(self, fps=25.0):
        self.fps = fps
        self.positions = {}  # player_id -> list of positions
        self.timestamps = {}  # player_id -> list of timestamps
        self.frame_count = 0
        
    def update_player_position(self, player_id, position):
        if player_id not in self.positions:
            self.positions[player_id] = []
            self.timestamps[player_id] = []
            
        self.positions[player_id].append(position)
        self.timestamps[player_id].append(self.frame_count / self.fps)
        self.frame_count += 1
        
    def get_player_speed(self, player_id):
        if player_id not in self.positions or len(self.positions[player_id]) < 2:
            return 0.0, 0.0
            
        positions = np.array(self.positions[player_id])
        timestamps = np.array(self.timestamps[player_id])
        
        # Calculate distances between consecutive positions
        distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        time_diffs = timestamps[1:] - timestamps[:-1]
        
        # Calculate speeds in m/s
        speeds = distances / time_diffs
        avg_speed = np.mean(speeds)
        
        # Convert to km/h
        speed_kmh = avg_speed * 3.6
        
        return avg_speed, speed_kmh
        
    def get_player_distance(self, player_id):
        if player_id not in self.positions or len(self.positions[player_id]) < 2:
            return 0.0
            
        positions = np.array(self.positions[player_id])
        distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        return np.sum(distances)
        
    def get_player_active_time(self, player_id):
        if player_id not in self.timestamps or len(self.timestamps[player_id]) == 0:
            return 0.0
        return self.timestamps[player_id][-1] - self.timestamps[player_id][0]

class CameraTracker:
    def __init__(self):
        self.prev_frame = None
        
    def estimate_camera_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None, None
            
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Estimate global movement
        movement = np.mean(flow, axis=(0, 1))
        
        # Calculate homography
        h, w = frame.shape[:2]
        y, x = np.mgrid[0:h:100, 0:w:100].reshape(2, -1)
        pts1 = np.vstack((x, y)).T
        pts2 = (pts1 + flow[::100, ::100].reshape(-1, 2))
        
        homography, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        
        self.prev_frame = gray
        return movement, homography

class PerspectiveTransformer:
    def __init__(self, field_width=105, field_height=68):
        self.field_width = field_width
        self.field_height = field_height
        self.transformation_matrix = None
        
    def estimate_field_points_from_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
            
        # Estimate field corners from line intersections
        h, w = frame.shape[:2]
        field_points = np.float32([
            [w * 0.1, h * 0.3],
            [w * 0.9, h * 0.3],
            [w * 0.1, h * 0.9],
            [w * 0.9, h * 0.9]
        ])
        
        return field_points
        
    def set_field_points(self, field_points):
        if field_points is None:
            return
            
        # Define real-world coordinates
        real_points = np.float32([
            [0, 0],
            [self.field_width, 0],
            [0, self.field_height],
            [self.field_width, self.field_height]
        ])
        
        self.transformation_matrix = cv2.getPerspectiveTransform(field_points, real_points)
        
    def transform_point(self, point):
        if self.transformation_matrix is None:
            return point
            
        transformed = cv2.perspectiveTransform(
            np.array([[point]], dtype=np.float32),
            self.transformation_matrix
        )
        return transformed[0][0]

def draw_results(frame, players, balls, team_labels, player_metrics, track_ids, team_colors):
    result = frame.copy()
    
    # Draw players
    for i, (player, track_id) in enumerate(zip(players, track_ids)):
        x1, y1, x2, y2 = map(int, player[:4])
        team_id = team_labels[i] if i < len(team_labels) else -1
        
        # Get color based on team
        if team_id == 0:
            color = (0, 0, 255)  # Red for team 1
        elif team_id == 1:
            color = (255, 0, 0)  # Blue for team 2
        else:
            color = (0, 255, 0)  # Green for referee/unknown
            
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw player ID and speed
        if track_id in player_metrics.positions:
            speed_ms, speed_kmh = player_metrics.get_player_speed(track_id)
            cv2.putText(
                result,
                f"ID: {track_id} ({speed_kmh:.1f} km/h)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
    # Draw balls
    for ball in balls:
        x1, y1, x2, y2 = map(int, ball[:4])
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
    return result

def process_video(input_path, output_path, model_path='yolov8n.pt', conf_threshold=0.25, field_width=105, field_height=68):
    """Process a video file and save the analysis results"""
    print(f"Processing video: {input_path}")
    
    # Initialize components
    detector = ObjectDetector(model_path=model_path)
    tracker = ObjectTracker()
    team_assigner = TeamAssigner()
    camera_tracker = CameraTracker()
    perspective_transformer = PerspectiveTransformer(
        field_width=field_width, 
        field_height=field_height
    )
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {input_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video opened successfully: {width}x{height} at {fps} FPS, {total_frames} total frames")
    
    # Initialize player metrics calculator
    player_metrics = PlayerMetrics(fps=fps)
    
    # Initialize video writer
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track IDs for players
    next_track_id = 0
    player_track_ids = {}
    
    # Process video frames
    frame_count = 0
    perspective_set = False
    team_colors = None
    
    # Store frame data and player stats
    frame_data_list = []
    player_stats_dict = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
            
        # Detect objects
        players, balls = detector.detect(frame, conf_threshold=conf_threshold)
        
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
        player_positions = []
        ball_positions = []
        
        for i, player in enumerate(players):
            x1, y1, x2, y2 = map(int, player[:4])
            player_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Simple tracking based on IoU
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
                
                # Store player position
                player_positions.append({
                    'track_id': best_id,
                    'team_id': team_labels[i] if i < len(team_labels) else -1,
                    'position': player_position_meters
                })
        
        # Store ball positions
        for ball in balls:
            x1, y1, x2, y2 = map(int, ball[:4])
            ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if perspective_set:
                ball_position_meters = perspective_transformer.transform_point(ball_center)
                ball_positions.append(ball_position_meters)
        
        # Store frame data
        frame_data_list.append({
            'frame_number': frame_count,
            'timestamp': frame_count / fps,
            'player_count': len(players),
            'ball_count': len(balls),
            'team1_count': sum(1 for t in team_labels if t == 0),
            'team2_count': sum(1 for t in team_labels if t == 1),
            'referee_count': sum(1 for t in team_labels if t == 2),
            'player_positions': player_positions,
            'ball_positions': ball_positions
        })
        
        # Draw results on frame
        result_frame = draw_results(
            frame.copy(), players, balls, team_labels, 
            player_metrics, track_ids, team_colors
        )
        
        # Display frame info
        cv2.putText(
            result_frame, 
            f"Frame: {frame_count} | Players: {len(players)} | Balls: {len(balls)}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Write output
        writer.write(result_frame)
        
        # Save sample frames for visualization (every 50 frames)
        if frame_count % 50 == 0:
            cv2.imwrite(f"frame_{frame_count}.jpg", result_frame)
                
        frame_count += 1
    
    # Clean up
    cap.release()
    writer.release()
    
    # Calculate final player statistics
    for track_id in player_track_ids.keys():
        speed_ms, speed_kmh = player_metrics.get_player_speed(track_id)
        distance = player_metrics.get_player_distance(track_id)
        active_time = player_metrics.get_player_active_time(track_id)
        
        # Find team ID for this player
        team_id = -1
        for frame_data in frame_data_list:
            for player_pos in frame_data['player_positions']:
                if player_pos['track_id'] == track_id:
                    team_id = player_pos['team_id']
                    break
            if team_id != -1:
                break
        
        player_stats_dict[track_id] = {
            'team_id': team_id,
            'total_distance': distance,
            'avg_speed': speed_kmh,
            'max_speed': speed_kmh,  # You might want to track max speed separately
            'active_time': active_time
        }
    
    print(f"Processing completed. {frame_count} frames processed.")
    print(f"Output saved to {output_path}")
    print(f"Sample frames saved as frame_X.jpg")
    
    return {
        'frame_data': frame_data_list,
        'player_stats': player_stats_dict,
        'total_frames': frame_count,
        'total_players': next_track_id,
        'fps': fps,
        'resolution': f"{width}x{height}"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Football Analysis System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--field-width', type=float, default=105, help='Football field width in meters')
    parser.add_argument('--field-height', type=float, default=68, help='Football field height in meters')
    parser.add_argument('--frames', type=int, default=0, help='Number of frames to process (0 for all)')
    args = parser.parse_args()
    
    process_video(
        args.video,
        args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        field_width=args.field_width,
        field_height=args.field_height
    ) 