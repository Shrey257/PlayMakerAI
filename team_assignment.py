import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self, n_clusters=3):
        """
        Initialize the team assigner with KMeans
        
        Args:
            n_clusters: Number of clusters for KMeans (typically 3: team1, team2, referee)
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.team_colors = None
        
    def extract_player_jersey_color(self, frame, bbox, padding=5):
        """
        Extract the dominant color from a player's jersey
        
        Args:
            frame: Input frame
            bbox: Bounding box of the player [x1, y1, x2, y2]
            padding: Padding to avoid including background
            
        Returns:
            dominant_color: The dominant color in BGR format
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Apply padding to focus on jersey (upper body)
        jersey_y1 = max(y1 + padding, 0)
        jersey_y2 = min(y1 + (y2 - y1) // 2, frame.shape[0])
        jersey_x1 = max(x1 + padding, 0)
        jersey_x2 = min(x2 - padding, frame.shape[1])
        
        # Extract jersey region
        jersey_roi = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        if jersey_roi.size == 0:
            return None
        
        # Convert to RGB for better color clustering
        jersey_roi_rgb = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2RGB)
        
        # Reshape to list of pixels
        pixels = jersey_roi_rgb.reshape(-1, 3).astype(np.float32)
        
        # Apply KMeans to find dominant color
        if len(pixels) > 0:
            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0].astype(np.uint8)
            
            # Convert back to BGR
            dominant_color_bgr = cv2.cvtColor(np.array([[dominant_color]]), cv2.COLOR_RGB2BGR)[0][0]
            return dominant_color_bgr
        
        return None
    
    def assign_teams(self, frame, player_detections):
        """
        Assign players to teams based on jersey colors
        
        Args:
            frame: Input frame
            player_detections: List of player detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            team_assignments: List of team assignments (0, 1, 2) for each player
        """
        jersey_colors = []
        valid_detections = []
        
        # Extract jersey colors for each player
        for detection in player_detections:
            color = self.extract_player_jersey_color(frame, detection)
            if color is not None:
                jersey_colors.append(color)
                valid_detections.append(detection)
        
        if not jersey_colors:
            return []
        
        # Cluster jersey colors to identify teams
        jersey_colors_array = np.array(jersey_colors)
        self.kmeans.fit(jersey_colors_array)
        
        # Get team labels
        team_labels = self.kmeans.labels_
        
        # Store team colors
        self.team_colors = self.kmeans.cluster_centers_
        
        return team_labels, valid_detections 