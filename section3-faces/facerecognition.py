import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = np.load('face_trained.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'../resources/Faces/train/Madonna/5.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, dist = face_recognizer.predict(faces_roi)

    #convert the confidence(dist) into percent
    similarity = max(0, min(1, (100 - dist) / 100))
    percent = similarity * 100

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.putText(img, f'Name: {people[label]}', (x, y - 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
    cv.putText(img, f'Conf: {percent:.2f}', (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
    cv.imshow('Detected Face', img)

cv.waitKey(0)
