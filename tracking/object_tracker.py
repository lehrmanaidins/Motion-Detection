from typing import Final
import numpy as np
import time
from .tracked_object import TrackedObject

class ObjectTracker:
    def __init__(
        self,
        distance_threshold: float = 10.0,
        max_display_lost_time: float = 5.0,
        max_lost_time: float = 30.0
    ):
        self.distance_threshold: Final[float] = distance_threshold
        self.max_display_lost_time: Final[float] = max_display_lost_time
        self.max_lost_time: Final[float] = max_lost_time

        self.tracked_objects: list[TrackedObject] = []
        self.lost_objects: list[TrackedObject] = []

    def _find_matching_tracked_object(
        self, detection: np.ndarray, object_list: list[TrackedObject]
    ) -> TrackedObject | None:
        """Find closest object within distance_threshold in a given list."""
        closest_object: TrackedObject | None = None
        closest_distance: float = float('inf')

        for obj in object_list:
            distance = np.linalg.norm(detection - obj.predicted_position)
            if distance < closest_distance and distance < self.distance_threshold:
                closest_distance = float(distance)
                closest_object = obj

        return closest_object

    def update(self, detections: list[tuple[float, float]]):
        current_time: Final[float] = time.time()

        # Predict positions for all objects (tracked + lost)
        for obj in self.tracked_objects + self.lost_objects:
            obj.update(new_position=None)

        # Track which objects are updated this frame
        updated_objects = set()  # store uuids of objects updated by a detection

        # Associate detections with tracked objects
        for detection in detections:
            detection_np = np.array(detection, dtype=np.float32)

            # Try to match with active tracked objects
            matched_obj = self._find_matching_tracked_object(detection_np, self.tracked_objects)
            if matched_obj:
                matched_obj.update(detection_np)
                updated_objects.add(matched_obj.uuid)
                continue

            # Try to match with lost objects
            matched_lost_obj = self._find_matching_tracked_object(detection_np, self.lost_objects)
            if matched_lost_obj:
                matched_lost_obj.update(detection_np)
                updated_objects.add(matched_lost_obj.uuid)
                self.lost_objects.remove(matched_lost_obj)
                self.tracked_objects.append(matched_lost_obj)
                continue

            # No match: create a new tracked object
            new_obj = TrackedObject(initial_position=detection_np)
            self.tracked_objects.append(new_obj)
            updated_objects.add(new_obj.uuid)

        # Move objects not updated this frame to lost_objects
        still_tracked = []
        for obj in self.tracked_objects:
            if obj.uuid in updated_objects:
                still_tracked.append(obj)
            else:
                # Object was not updated -> recently lost
                self.lost_objects.append(obj)
        self.tracked_objects = still_tracked

        # Remove lost objects that exceeded max_lost_time
        self.lost_objects = [
            obj for obj in self.lost_objects
            if current_time - obj.time_last_seen < self.max_lost_time
        ]

        # Return all objects for display purposes
        display_objects = self.tracked_objects.copy()
        for obj in self.lost_objects:
            if current_time - obj.time_last_seen <= self.max_display_lost_time:
                display_objects.append(obj)

        return display_objects
