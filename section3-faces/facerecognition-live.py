import numpy as np
import cv2 as cv
import os

def load_resources():
    required_files = [
        'haar_face.xml',
        'face_trained.npy',
        'labels.npy',
        'face_trained.yml'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found")

    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    features = np.load('face_trained.npy', allow_pickle=True)
    labels = np.load('labels.npy')
    
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')
    
    return haar_cascade, features, labels, face_recognizer

def main():
    try:
        haar_cascade, features, labels, face_recognizer = load_resources()
        
        people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 
                 'Madonna', 'Mindy Kaling', 'Peggy Schneider']

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            try:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y + h, x:x + w]
                    label, dist = face_recognizer.predict(faces_roi)

                    similarity = max(0, min(1, (100 - dist) / 100))
                    percent = similarity * 100

                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                    cv.putText(frame, f'Name: {people[label]}', (x, y - 30), 
                             cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                    cv.putText(frame, f'Conf: {percent:.2f}%', (x, y - 10), 
                             cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)

                cv.imshow('Face Recognition', frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()