import cv2
import numpy as np
from keras.models import load_model
from collections import Counter
import os

# Load model globally once
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'mobilenetv2.keras')
model = load_model(MODEL_PATH)
emotion_labels = ['angry', 'happy', 'sad', 'surprise', 'neutral']
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion_image(img_path):
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detectMultiScale(rgb, 1.1, 5)
    for (x, y, w, h) in faces:
        face_crop = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (128,128))
        face_input = np.expand_dims(face_resized, axis=0)/255.0
        pred = model.predict(face_input, verbose=0)
        return emotion_labels[np.argmax(pred)]
    return "No Face Detected"

def predict_emotion_webcam():
    cap = cv2.VideoCapture(0)
    emotions_detected = []
    total_frames = 20
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(rgb, 1.1, 5)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (224,224))
            face_input = np.expand_dims(face_resized, axis=0)/255.0
            pred = model.predict(face_input, verbose=0)
            emotions_detected.append(emotion_labels[np.argmax(pred)])
        frame_count += 1

    cap.release()
    if emotions_detected:
        majority_emotion = Counter(emotions_detected).most_common(1)[0][0]
        return majority_emotion
    return "No Face Detected"