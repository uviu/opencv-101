import cv2 as cv
import os

# Get the absolute path to the image
image_path = 'resources/Videos/dog.mp4'

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: Video file not found at '{image_path}'")
    exit(1)

# Read the image
capture = cv.VideoCapture(image_path)

# Check if image was successfully loaded
if capture is None:
    print(f"Error: Could not load video from '{image_path}'")
    exit(1)

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
