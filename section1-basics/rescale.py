import cv2 as cv
import os

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #only applicable for live video
    capture.set(3, width)
    capture.set(4, height)


#video_path = 'resources/Videos/dog.mp4'
video_path = 0

if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'")
    exit(1)

capture = cv.VideoCapture(video_path)

if capture is None:
    print(f"Error: Could not load video from '{video_path}'")
    exit(1)

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
    cv.imshow('Video', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
