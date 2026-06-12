Emotion-Music-Recommender/
│
├── backend/
│   ├── main.py                  # FastAPI backend server
│   ├── music_recommender.py     # Emotion-to-music mapping logic
│   ├── text_models.py           # NLP-based emotion detection
│   ├── image_models.py          # CNN-based facial emotion detection
│
├── models/
│   └── vgg16+transformer.keras  # Trained 5-class facial emotion model
│
├── data/
│   └── Music Info.csv           # Song metadata (valence, energy, etc.)
│
├── frontend/
│   └── app.py                   # Streamlit web UI
│
├── requirements.txt             # Project dependencies
└── README.md                    # You are here!


## How to Run the Emotion → Music Recommender Application

# Create a virtual environment:
python -m venv .venv

# Activate the virtual environment:
- For Windows (CMD):
.venv\Scripts\activate
- For Mac or Linux:
source .venv/bin/activate

# Install required dependencies:
pip install -r requirements.txt

# Run the FastAPI backend:
uvicorn backend.main:app --reload

(The backend will run at http://localhost:8000)

# In a new terminal (keep backend running), run the Streamlit frontend:
streamlit run frontend/apps.py

(The frontend will run at http://localhost:8501)