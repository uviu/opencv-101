import cv2 as cv
import os

# Get the absolute path to the image
image_path = '../resources/Photos/cat_large.jpg'

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at '{image_path}'")
    exit(1)

# Read the image
img = cv.imread(image_path)

# Check if image was successfully loaded
if img is None:
    print(f"Error: Could not load image from '{image_path}'")
    exit(1)

cv.imshow('Cat', img)
cv.waitKey(0)