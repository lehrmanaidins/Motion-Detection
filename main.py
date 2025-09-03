
from typing import Final
import itertools
import numpy as np
import cv2
from youtube_link import get_youtube_stream_url

def main():
    # live_stream_url: Final[str] = 'https://www.youtube.com/watch?v=FnQJ65mL0dE' # LIVE, Stars, Meteors, Aurora, from Dark Sky Maine - Chill Relax Star-watching Music CAM 1
    # live_stream_url: Final[str] = 'https://www.youtube.com/watch?v=Xe2g_54uj_U' # Stars, Aurora, Meteors LIVE Cam from Mt Katahdin from MAINE US - CAM 2
    live_stream_url: Final[str] = 'https://www.youtube.com/watch?v=H999s0P1Er0' # Live High-Definition Views from the International Space Station (Official NASA Stream)
    # live_stream_url: Final[str] = 'https://www.youtube.com/watch?v=DnUFAShZKus' # Peace Bridge - Canada Bound

    # Get the direct stream URL
    stream_url: Final[str] = get_youtube_stream_url(live_stream_url)
    capture = cv2.VideoCapture(stream_url)

    if not capture.isOpened():
        print("Error: Could not open video stream.")
        return

    previous_frame = None
    stall_count = 0
    stall_limit: Final[int] = 2  # Number of nearly-identical frames before considering it a stall

    while True:
        ret, current_frame = capture.read()
        if not ret:
            print("Stream ended or interrupted (read returned False).")
            break

        if previous_frame is not None:
            # Use mean squared error to detect nearly-identical frames
            '''error = np.mean((current_frame.astype("float") - previous_frame.astype("float")) ** 2)
            if error < 1:  # Adjust threshold as needed
                stall_count += 1
                if stall_count > stall_limit:
                    print("Stream appears to be stalled (frames nearly identical).")
                    break
            else:
                stall_count = 0
'''
            # Show frame difference
            diff = cv2.absdiff(previous_frame, current_frame)
            cv2.imshow("Frame Difference", diff)

            # Show difference multiplied by current frame (highlights motion)
            diff_times_current_frame = cv2.multiply(current_frame, cv2.divide(diff, 255.0))
            cv2.imshow("Diff Times Current Frame", diff_times_current_frame)

        cv2.imshow("Original Frame", current_frame)
        previous_frame = current_frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', filename='profile_output.txt', sort='cumtime')