import cv2
import numpy as np

class CameraTracker:
    def __init__(self):
        """
        Initialize the camera tracker with optical flow parameters
        """
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Create random colors for visualization
        self.colors = np.random.randint(0, 255, (100, 3))
        
        # First frame and previous points placeholder
        self.prev_frame = None
        self.prev_points = None
        self.homography = None
        
    def estimate_camera_movement(self, frame):
        """
        Estimate camera movement between frames using optical flow
        
        Args:
            frame: Current frame
            
        Returns:
            camera_movement: Estimated (dx, dy) camera movement
            homography: Homography matrix for perspective transformation
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If this is the first frame, initialize
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return (0, 0), np.eye(3)
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_points, None, **self.lk_params
        )
        
        # Select good points
        if next_points is not None:
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            # Calculate movement
            if len(good_new) > 0 and len(good_old) > 0:
                # Calculate homography matrix if enough points
                if len(good_new) >= 4:
                    H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                    self.homography = H
                else:
                    H = self.homography if self.homography is not None else np.eye(3)
                
                # Calculate average movement
                movement = good_new - good_old
                avg_movement = np.mean(movement, axis=0)
                
                # Update previous points and frame
                self.prev_points = good_new.reshape(-1, 1, 2)
                self.prev_frame = gray
                
                return tuple(avg_movement), H
        
        # If no movement detected, refresh feature points
        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self.prev_frame = gray
        
        return (0, 0), np.eye(3) if self.homography is None else self.homography 