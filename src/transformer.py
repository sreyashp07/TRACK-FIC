import cv2
import numpy as np


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initializes a perspective transformer from source points to target points.

        Args:
            source (np.ndarray): Source coordinates (4x2) in the original view.
            target (np.ndarray): Target coordinates (4x2) in the transformed view.
        """
        self.matrix = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Applies perspective transformation to a set of 2D points.

        Args:
            points (np.ndarray): Array of (x, y) points to transform.

        Returns:
            np.ndarray: Transformed (x, y) points.
        """

        # Handle empty input safely
        if points is None or len(points) == 0:
            return points

        # Reshape for OpenCV format: (N, 1, 2)
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)

        # Apply perspective transformation
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)

        # Restore shape to (N, 2)
        return transformed_points.reshape(-1, 2)
