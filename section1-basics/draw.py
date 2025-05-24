import cv2 as cv
import numpy as np

#create dummy blank-image
blank = np.zeros((500, 500, 3), dtype='uint8')
#cv.imshow('Blank', blank)

blank[:] = 125, 255, 125

blank[200:300, 300:400] = 0, 0, 255

cv.rectangle(blank, (100, 100), (200, 200), (255, 0, 0), thickness=2)

cv.rectangle(blank, (200, 200), ((blank.shape[1]//2), blank.shape[0]//2), (255, 0, 0), thickness=cv.FILLED)
#cv.imshow('Rectangle', blank)

cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=3)
cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=3)

cv.putText(blank, 'testing', (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (123, 45, 43), 2)
cv.imshow('Text', blank)

cv.waitKey(0)