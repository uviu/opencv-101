import cv2 as cv
import os

# Get the absolute path to the image
video_path = '../resources/Videos/dog.mp4'

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'")
    exit(1)

# Read the image
capture = cv.VideoCapture(video_path)

# Check if image was successfully loaded
if capture is None:
    print(f"Error: Could not load video from '{video_path}'")
    exit(1)

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
