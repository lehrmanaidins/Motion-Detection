
from typing import Final
import cv2
import numpy as np

def main() -> None:
    source_video_file_path: Final[str] = 'night_sky_140min.mp4'

    capture = cv2.VideoCapture(source_video_file_path)

    if not capture.isOpened():
        print("Error: Could not open video file.")
        return

    alpha: Final[float] = 0.1
    averaged_frame: np.ndarray | None = None
    bitwise_or_frame: np.ndarray | None = None

    while True:
        ret, current_frame = capture.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        if averaged_frame is None:
            # Initialize as float32
            averaged_frame = current_frame.astype(np.float32)
            continue

        if bitwise_or_frame is None:
            bitwise_or_frame = current_frame.copy()

        cv2.accumulateWeighted(current_frame, averaged_frame, alpha)
        bitwise_or_frame = cv2.bitwise_or(bitwise_or_frame, current_frame)

        cv2.imshow("Current Frame", current_frame)
        cv2.imshow("Averaged Frame", cv2.convertScaleAbs(averaged_frame))
        cv2.imshow("Bitwise OR Frame", bitwise_or_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
