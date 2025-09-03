
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import sys
import cv2
import numpy as np
from tracking.object_tracker import ObjectTracker

class SatelliteViewer(QMainWindow):
    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle("Satellite Tracker")

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file.")

        # Tracker
        self.object_tracker = ObjectTracker(max_lost=10, distance_threshold=10)
        self.averaged_frame = None
        self.alpha = 0.01

        # UI elements
        self.video_label = QLabel()
        self.trail_btn = QPushButton("Toggle Trails")
        self.predict_btn = QPushButton("Toggle Predictions")
        self.labels_btn = QPushButton("Toggle Labels")

        self.show_trails = True
        self.show_predictions = True
        self.show_labels = True

        self.trail_btn.clicked.connect(self.toggle_trails)
        self.predict_btn.clicked.connect(self.toggle_predictions)
        self.labels_btn.clicked.connect(self.toggle_labels)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.trail_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.labels_btn)
        layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Object history
        self.history = {}

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

    def toggle_trails(self):
        self.show_trails = not self.show_trails

    def toggle_predictions(self):
        self.show_predictions = not self.show_predictions

    def toggle_labels(self):
        self.show_labels = not self.show_labels

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.timer.stop()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.averaged_frame is None:
            self.averaged_frame = gray.astype(np.float32)
        else:
            cv2.accumulateWeighted(gray, self.averaged_frame, self.alpha)

        background_display = cv2.convertScaleAbs(self.averaged_frame)
        frame_delta = cv2.absdiff(background_display, gray)
        frame_delta_corrected = cv2.convertScaleAbs(frame_delta, alpha=1.0, beta=-75)
        _, motion_mask = cv2.threshold(frame_delta_corrected, 40, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            detections.append((cx, cy))

        objects = self.object_tracker.update(detections)

        contour_frame = frame.copy()
        for obj in objects:
            object_id = str(obj["object_id"])[:8]  # shorten for display
            cx, cy = map(int, obj["centroid"])

            # Save history
            if object_id not in self.history:
                self.history[object_id] = []
            self.history[object_id].append((cx, cy))
            if len(self.history[object_id]) > 50:
                self.history[object_id].pop(0)

            # Draw trails
            if self.show_trails:
                for i in range(1, len(self.history[object_id])):
                    cv2.line(contour_frame, self.history[object_id][i-1],
                             self.history[object_id][i], (0, 0, 255), 1, cv2.LINE_AA)

            # Draw predicted position
            if self.show_predictions:
                # Simple linear prediction: next_pos = 2*current - previous
                if len(self.history[object_id]) >= 2:
                    prev = np.array(self.history[object_id][-2])
                    curr = np.array(self.history[object_id][-1])
                    pred = 2*curr - prev
                    pred = tuple(pred.astype(int))
                    cv2.line(contour_frame, (cx, cy), pred, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(contour_frame, pred, 3, (0, 255, 0), -1)

            # Draw labels
            if self.show_labels:
                label = f"id:{object_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                top_left = (cx + 10, cy - 10 - text_height)
                bottom_right = (cx + 10 + text_width, cy - 10 + baseline)
                cv2.rectangle(contour_frame, top_left, bottom_right, (0, 0, 0), cv2.FILLED)
                cv2.putText(contour_frame, label, (cx + 10, cy - 10), font, font_scale, (255, 255, 255), thickness)

        # Convert to QImage and show
        rgb_image = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SatelliteViewer("night_sky_140min.mp4")
    window.show()
    sys.exit(app.exec())
