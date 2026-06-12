from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from backend.text_models import predict_text_emotion

# =========================
# CONFIG
# =========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2.keras"
MUSIC_DATA_PATH = BASE_DIR / "data" / "Music Info.csv"
# MODEL_PATH = r"E:\internships\infosys springboard\Emotion-app\Emotion-app\models\mobilenetv2.keras"
# MUSIC_DATA_PATH = r"E:\internships\infosys springboard\Emotion-app\Emotion-app\data\Music Info.csv"

EMOTION_LABELS = ['angry', 'happy', 'sad', 'surprise', 'neutral']

# =========================
# APP INIT
# =========================

app = FastAPI(title="Mood Mate Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Model to handle clean JSON requests from your Streamlit frontend
class TextInput(BaseModel):
    text: str

# =========================
# LOAD MODEL
# =========================

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ MobileNetV2 emotion model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

# =========================
# LOAD MUSIC DATA
# =========================

try:
    songs_df = pd.read_csv(MUSIC_DATA_PATH)
    songs_df.rename(columns={"name": "track_name", "id": "spotify_id"}, inplace=True)
    print("✅ Music database loaded successfully")
except Exception as e:
    print("⚠️ Music CSV not loaded:", e)
    songs_df = pd.DataFrame()

# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# HELPERS
# =========================

def get_recommendations(emotion: str):
    if songs_df.empty:
        return []

    ranges = {
        'happy': (0.7, 1.0),
        'sad': (0.0, 0.3),
        'angry': (0.3, 0.5),
        'surprise': (0.6, 0.9),
        'neutral': (0.4, 0.6),
    }

    vmin, vmax = ranges.get(emotion, (0.4, 0.6))
    filtered = songs_df[
        (songs_df["valence"] >= vmin) & (songs_df["valence"] <= vmax)
    ]

    if filtered.empty:
        return []

    samples = filtered.sample(min(4, len(filtered)))

    return [
        {
            "name": r.track_name,
            "artist": r.artist,
            "link": f"https://open.spotify.com/track/{r.spotify_id}"
        }
        for r in samples.itertuples()
    ]


def process_image(contents: bytes):
    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file structure received")

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise ValueError("No face detected in the frame. Please look directly at the camera.")

    # Take largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # Preprocessing to match MobileNetV2 inputs
    face = cv2.resize(face, (128, 128))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")

    # Preprocess values according to the specific model architecture 
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face

# =========================
# ROUTES
# =========================

@app.get("/")
async def root():
    return {
        "status": "online",
        "model_loaded": model is not None
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "MobileNetV2 Model not loaded on the server"})

    try:
        contents = await file.read()
        img = process_image(contents)

        preds = model.predict(img, verbose=0)[0]

        # Debug logs
        print("🔍 Deep Learning Predictions:", dict(zip(EMOTION_LABELS, preds.round(3))))

        idx = int(np.argmax(preds))
        emotion = EMOTION_LABELS[idx]
        confidence = float(preds[idx])

        return {
            "emotion": emotion,
            "confidence": f"{confidence:.2%}",
            "songs": get_recommendations(emotion)
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

# Kept this to make sure if the frontend still hits /predict/face-image, it doesn't crash with a 404
@app.post("/predict/face-image")
async def predict_face_image_alias(file: UploadFile = File(...)):
    return await predict_image(file)


@app.post("/predict/text")
async def predict_text(payload: TextInput):
    try:
        

        # Run NLP prediction logic
        emotion = predict_text_emotion(payload.text).lower()
        return {
            "emotion": emotion,
            "confidence": "NLP Derived",  # Added to prevent missing UI elements
            "songs": get_recommendations(emotion)
        }

    except Exception as e:
        print("❌ TEXT PROCESSING ERROR:", e)
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )