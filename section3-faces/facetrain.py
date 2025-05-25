import cv2 as cv
import os
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Peggy Schneider']
DIR = r'../resources/Faces/train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        print(f'Processing {person} images...')
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
label = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#train recognizer on features and labels
face_recognizer.train(features, np.array(label))

face_recognizer.save('face_trained.yml')
np.save('face_trained.npy', features)
np.save('labels.npy', label)