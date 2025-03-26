import cv2
import numpy as np
from collections import defaultdict

class PlayerMetrics:
    def __init__(self, fps=25, smoothing_window=5):
        """
        Initialize the player metrics calculator
        
        Args:
            fps: Frames per second of the video
            smoothing_window: Window size for smoothing positions
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.player_positions = defaultdict(list)  # track_id -> list of positions
        self.player_speeds = defaultdict(list)  # track_id -> list of speeds
        self.player_distances = defaultdict(float)  # track_id -> total distance
        
    def update_player_position(self, track_id, position_meters):
        """
        Update a player's position
        
        Args:
            track_id: Unique ID for the player
            position_meters: Position in meters (x, y)
        """
        self.player_positions[track_id].append(position_meters)
        
        # Limit the history to smoothing window size
        if len(self.player_positions[track_id]) > self.smoothing_window:
            self.player_positions[track_id].pop(0)
            
        # Calculate speed if we have at least 2 positions
        if len(self.player_positions[track_id]) >= 2:
            pos1 = self.player_positions[track_id][-2]
            pos2 = self.player_positions[track_id][-1]
            
            # Calculate distance in meters
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Calculate speed in m/s
            speed = distance * self.fps
            self.player_speeds[track_id].append(speed)
            
            # Update total distance
            self.player_distances[track_id] += distance
            
            # Limit the speed history to smoothing window size
            if len(self.player_speeds[track_id]) > self.smoothing_window:
                self.player_speeds[track_id].pop(0)
                
    def get_player_speed(self, track_id):
        """
        Get a player's current speed (smoothed)
        
        Args:
            track_id: Unique ID for the player
            
        Returns:
            speed: Current speed in m/s and km/h
        """
        if track_id not in self.player_speeds or not self.player_speeds[track_id]:
            return 0, 0
            
        # Calculate smoothed speed
        smoothed_speed_ms = np.mean(self.player_speeds[track_id])
        smoothed_speed_kmh = smoothed_speed_ms * 3.6
        
        return smoothed_speed_ms, smoothed_speed_kmh
    
    def get_player_distance(self, track_id):
        """
        Get a player's total distance covered
        
        Args:
            track_id: Unique ID for the player
            
        Returns:
            distance: Total distance in meters
        """
        return self.player_distances.get(track_id, 0)
    
    def reset(self):
        """
        Reset all player data
        """
        self.player_positions.clear()
        self.player_speeds.clear()
        self.player_distances.clear() 