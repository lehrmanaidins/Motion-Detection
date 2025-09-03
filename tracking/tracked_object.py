
from typing import Final
import uuid
import secrets
import time
import numpy as np

class TrackedObject:
    def __init__(self, initial_position: np.ndarray):

        self.uuid: Final[str] = str(uuid.uuid4())
        self.display_id: Final[str] = secrets.token_hex(3) # 6 hex digits

        self.last_known_position: np.ndarray = initial_position
        self.predicted_position: np.ndarray = initial_position

        self.time_discovered: Final[float] = time.time()
        self.time_last_seen: float = time.time()

        self.position_history: list[tuple[float, np.ndarray]] = [(self.time_discovered, self.last_known_position)]

    def update(self, new_position: np.ndarray | None):
        if new_position is None:
            self.predicted_position = self.predict_next_position()
            return

        self.last_known_position = new_position
        self.predicted_position = new_position

        self.time_last_seen = time.time()

        self.position_history.append(
            (self.time_last_seen, new_position)
        )

    def predict_next_position(self) -> np.ndarray:
        velocity: np.ndarray = self.get_current_velocity()
        time_since_last_seen: float = time.time() - self.time_last_seen
        self.predicted_position = self.last_known_position + velocity * time_since_last_seen
        return self.predicted_position
    
    def get_current_velocity(self) -> np.ndarray:
        if len(self.position_history) < 2:
            return np.array([0.0, 0.0])
        
        velocities: list[np.ndarray] = []

        max_number_of_previous_positions_to_average: Final[int] = 20
        n = min(max_number_of_previous_positions_to_average, len(self.position_history) - 1)

        for i in range(1, n + 1):
            t1, p1 = self.position_history[-i - 1]
            t2, p2 = self.position_history[-i]
            dt = t2 - t1
            if dt == 0:
                continue
            velocity = (p2 - p1) / dt
            velocities.append(velocity)

        if not velocities:
            return np.array([0.0, 0.0])

        average_velocity: np.ndarray = np.sum(velocities, axis=0) / len(velocities)
        return average_velocity
