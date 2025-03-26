import cv2
import numpy as np
import matplotlib.pyplot as plt

class PerspectiveTransformer:
    def __init__(self, field_width=105, field_height=68):
        """
        Initialize the perspective transformer
        
        Args:
            field_width: Width of the football field in meters
            field_height: Height of the football field in meters
        """
        self.field_width = field_width
        self.field_height = field_height
        self.perspective_matrix = None
        self.field_points = None
        self.image_points = None
        
    def set_field_points(self, field_points):
        """
        Set the field points (corners of the field in the image)
        
        Args:
            field_points: List of 4 points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        self.field_points = np.array(field_points, dtype=np.float32)
        
        # Target points for perspective transformation (top-down view)
        # Corresponds to the field corners in meters
        self.target_points = np.array([
            [0, 0],
            [self.field_width, 0],
            [self.field_width, self.field_height],
            [0, self.field_height]
        ], dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        self.perspective_matrix = cv2.getPerspectiveTransform(self.field_points, self.target_points)
        
    def transform_point(self, point):
        """
        Transform a point from image coordinates to field coordinates (meters)
        
        Args:
            point: Point in image coordinates (x, y)
            
        Returns:
            transformed_point: Point in field coordinates (meters)
        """
        if self.perspective_matrix is None:
            raise ValueError("Perspective matrix not set. Call set_field_points first.")
        
        # Convert to homogeneous coordinates
        point_h = np.array([[point[0], point[1], 1]], dtype=np.float32)
        
        # Apply the transformation
        transformed = np.dot(self.perspective_matrix, point_h.T)
        
        # Convert back from homogeneous coordinates
        transformed = transformed / transformed[2]
        
        return (transformed[0][0], transformed[1][0])
    
    def estimate_field_points_from_lines(self, frame):
        """
        Estimate field points from detected lines in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            field_points: Estimated field points
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # This is a simplified implementation - a real one would need more complex
        # line processing to identify field boundaries correctly
        # For demonstration, we'll just return the image corners
        h, w = frame.shape[:2]
        field_points = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        return field_points
        
    def visualize_transformation(self, frame):
        """
        Visualize the perspective transformation
        
        Args:
            frame: Input frame
            
        Returns:
            visualization: Visualization image
        """
        if self.perspective_matrix is None:
            raise ValueError("Perspective matrix not set. Call set_field_points first.")
        
        # Create a warped image (top-down view)
        h, w = frame.shape[:2]
        warped = cv2.warpPerspective(frame, self.perspective_matrix, (int(self.field_width*10), int(self.field_height*10)))
        
        # Draw points on the original image
        viz_original = frame.copy()
        for point in self.field_points:
            cv2.circle(viz_original, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        
        # Draw lines between points
        for i in range(4):
            pt1 = (int(self.field_points[i][0]), int(self.field_points[i][1]))
            pt2 = (int(self.field_points[(i+1)%4][0]), int(self.field_points[(i+1)%4][1]))
            cv2.line(viz_original, pt1, pt2, (0, 255, 255), 2)
        
        # Create a combined visualization
        viz = np.hstack([viz_original, cv2.resize(warped, (w, h))])
        
        return viz 