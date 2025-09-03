
from typing import Final
import time
import cv2
import numpy as np

from tracking import ObjectTracker, TrackedObject

def main() -> None:
    source_video_file_path: Final[str] = 'night_sky_140min.mp4'

    capture = cv2.VideoCapture(source_video_file_path)

    if not capture.isOpened():
        print("Error: Could not open video file.")
        return

    alpha: Final[float] = 0.005  # How fast to update the background for static regions
    averaged_frame: np.ndarray | None = None

    object_tracker = ObjectTracker(
        distance_threshold=10,  # pixels
    )

    while True:
        ret, current_frame = capture.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Convert to grayscale
        current_frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if averaged_frame is None:
            # Initialize as float32
            averaged_frame = current_frame_grayscale.astype(np.float32)
            continue

        # Convert running average to uint8 for display/comparison
        background_display = cv2.convertScaleAbs(averaged_frame)

        # Compute difference to detect motion
        frame_delta = cv2.absdiff(background_display, current_frame_grayscale)
        star_mask = cv2.threshold(background_display, 87.5, 255, cv2.THRESH_BINARY)[1]
        star_mask = cv2.dilate(star_mask, None, iterations=2)
        ignore_mask = cv2.bitwise_not(star_mask)
        frame_delta = cv2.bitwise_and(frame_delta, frame_delta, mask=ignore_mask)

        frame_delta_corrected = cv2.convertScaleAbs(frame_delta, alpha=1.0, beta=-75)
        frame_delta_corrected = cv2.threshold(frame_delta_corrected, 100, 255, cv2.THRESH_TOZERO)[1]
        # frame_delta_corrected = cv2.dilate(frame_delta_corrected, None, iterations=1)
        # frame_delta_corrected = cv2.erode(frame_delta_corrected, None, iterations=1)

        # Threshold to detect moving objects (satellites)
        motion_thresh = 40
        _, motion_mask = cv2.threshold(frame_delta_corrected, motion_thresh, 255, cv2.THRESH_BINARY)

        # Update averaged frame slowly
        cv2.accumulateWeighted(current_frame_grayscale, averaged_frame, alpha)

        # Draw contours for visualization
        contour_frame = current_frame.copy()
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            detections.append((cx, cy))

        object_tracker.update(detections)
        tracked_objects: list[TrackedObject] = object_tracker.tracked_objects

        # Draw active tracked objects
        for tracked_object in object_tracker.tracked_objects:
            object_display_id = tracked_object.display_id
            object_centroid = tracked_object.predicted_position
            object_position_history = tracked_object.position_history

            cx, cy = map(int, object_centroid)
            offset_x, offset_y = 10, -10
            dot_pos = (cx + offset_x, cy + offset_y)

            # Draw line from centroid to dot 
            cv2.line(contour_frame, (cx + 5, cy - 5), dot_pos, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Text label in white for tracked objects
            label_color = (255, 255, 255)
            label = f'{object_display_id}'
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.35
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            top_left = (dot_pos[0] + 5, dot_pos[1] - 5 - text_height)
            bottom_right = (dot_pos[0] + 5 + text_width, dot_pos[1] - 5 + baseline)

            cv2.rectangle(contour_frame, top_left, bottom_right, (0, 0, 0), cv2.FILLED)
            cv2.putText(contour_frame, label, (dot_pos[0] + 5, dot_pos[1] - 5), font, font_scale, label_color, thickness)

            # Trail color based on hex display ID
            try:
                r = int(object_display_id[0:2], 16)
                g = int(object_display_id[2:4], 16)
                b = int(object_display_id[4:6], 16)
                trail_color = (b, g, r)
            except ValueError:
                trail_color = (255, 255, 255)

            for i in range(1, len(object_position_history)):
                pos_prev = object_position_history[i-1][1]
                pos_curr = object_position_history[i][1]
                cv2.line(contour_frame, tuple(map(int, pos_prev)), tuple(map(int, pos_curr)), trail_color, 1, cv2.LINE_AA)

        current_time = time.time()
        for lost_object in object_tracker.lost_objects:
            # Only display if recently lost
            time_since_lost = current_time - lost_object.time_last_seen
            if time_since_lost > object_tracker.max_display_lost_time:
                continue  # skip, too old

            object_display_id = lost_object.display_id
            object_centroid = lost_object.predicted_position
            object_position_history = lost_object.position_history

            cx, cy = map(int, object_centroid)
            offset_x, offset_y = 10, -10
            dot_pos = (cx + offset_x, cy + offset_y)

            # Draw line from centroid to dot 
            cv2.line(contour_frame, (cx + 5, cy - 5), dot_pos, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Fade label color from white to red
            fade_ratio = min(time_since_lost / object_tracker.max_display_lost_time, 1.0)
            r = 255
            g = int(255 * (1 - fade_ratio))
            b = int(255 * (1 - fade_ratio))
            label_color = (b, g, r)

            label = f'{object_display_id}'

            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.35
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            top_left = (dot_pos[0] + 5, dot_pos[1] - 5 - text_height)
            bottom_right = (dot_pos[0] + 5 + text_width, dot_pos[1] - 5 + baseline)

            cv2.rectangle(contour_frame, top_left, bottom_right, (0, 0, 0), cv2.FILLED)
            cv2.putText(contour_frame, label, (dot_pos[0] + 5, dot_pos[1] - 5), font, font_scale, label_color, thickness)

            # Draw trail (unchanged)
            try:
                r_t = int(object_display_id[0:2], 16)
                g_t = int(object_display_id[2:4], 16)
                b_t = int(object_display_id[4:6], 16)
                trail_color = (b_t, g_t, r_t)
            except ValueError:
                trail_color = (255, 255, 255)

            for i in range(1, len(object_position_history)):
                pos_prev = object_position_history[i-1][1]
                pos_curr = object_position_history[i][1]
                cv2.line(contour_frame, tuple(map(int, pos_prev)), tuple(map(int, pos_curr)), trail_color, 1, cv2.LINE_AA)

        # Show frames
        # cv2.imshow("Orininal Video", current_frame)
        # cv2.imshow("Background (Long-term Avg)", background_display)
        cv2.imshow("Frame Delta", frame_delta)
        cv2.imshow("Star Mask", star_mask)
        cv2.imshow("Ignore Mask", ignore_mask)
        cv2.imshow("Masked Background", cv2.bitwise_and(background_display, background_display, mask=ignore_mask))
        # cv2.imshow("Frame Delta Corrected", frame_delta_corrected)
        # cv2.imshow("Motion Mask", motion_mask)
        cv2.imshow("Satellite Tracking", contour_frame)
        # cv2.imshow("Highlighted Motion", cv2.bitwise_and(current_frame, current_frame, mask=motion_mask))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
