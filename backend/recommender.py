import pandas as pd
import numpy as np
import os

# ======================================================
# 1️⃣ Load Dataset
# ======================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Music Info.csv')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add 'Music Info.csv'.")

music_df = pd.read_csv(DATA_PATH)

# ======================================================
# 2️⃣ Add Spotify URL column
# ======================================================
def track_id_to_spotify_url(track_id):
    base_url = "https://open.spotify.com/track/"
    return f"{base_url}{track_id}"

music_df["spotify_url"] = music_df["spotify_id"].apply(track_id_to_spotify_url)

# ======================================================
# 3️⃣ Core Recommendation Function (Supports Uplift Mode)
# ======================================================
def recommend_songs_by_emotion(emotion: str, n: int = 5, uplift: bool = False):
    """
    Recommend songs based on emotion.
    - uplift=False → Mood-matching mode (songs that reflect your emotion)
    - uplift=True  → Mood-uplifting mode (songs that improve your emotion)
    """

    emotion = emotion.lower().strip()

    # ------------------------------
    # Mood-Matching Mode
    # ------------------------------
    if not uplift:
        if emotion == "sad":
            recs = music_df[(music_df["valence"] < 0.4) & (music_df["energy"] < 0.5)]
        elif emotion == "happy":
            recs = music_df[(music_df["valence"] > 0.6) & (music_df["energy"] > 0.5)]
        elif emotion == "angry":
            recs = music_df[(music_df["valence"] < 0.4) & (music_df["energy"] > 0.7)]
        elif emotion == "surprised":
            recs = music_df[(music_df["valence"].between(0.4, 0.7)) & (music_df["energy"] > 0.6)]
        elif emotion == "fearful":
            recs = music_df[(music_df["valence"] < 0.4) & (music_df["energy"].between(0.6, 1.0))]
        else:  # neutral or unknown
            recs = music_df[(music_df["valence"].between(0.4, 0.6)) & (music_df["energy"].between(0.4, 0.6))]

    # ------------------------------
    # Mood-Uplifting Mode
    # ------------------------------
    else:
        if emotion == "sad":
            # Sad → Play positive, mid-energy tracks
            recs = music_df[(music_df["valence"] > 0.6) & (music_df["energy"].between(0.4, 0.7))]
        elif emotion == "angry":
            # Angry → Play calm, positive songs
            recs = music_df[(music_df["valence"] > 0.5) & (music_df["energy"] < 0.5)]
        elif emotion == "fearful":
            # Fear → Play soothing and pleasant tracks
            recs = music_df[(music_df["valence"] > 0.6) & (music_df["energy"].between(0.3, 0.6))]
        elif emotion == "happy":
            # Happy → Maintain good vibes
            recs = music_df[(music_df["valence"] > 0.6) & (music_df["energy"] > 0.5)]
        elif emotion == "surprised":
            # Surprise → Moderate excitement
            recs = music_df[(music_df["valence"].between(0.5, 0.8)) & (music_df["energy"] > 0.6)]
        else:
            # Neutral → Balanced tracks
            recs = music_df[(music_df["valence"].between(0.4, 0.6)) & (music_df["energy"].between(0.4, 0.6))]

    # Fallback if no results
    if recs.empty:
        recs = music_df.sample(n)

    # Randomly sample results
    recs_sampled = recs.sample(min(n, len(recs)))

    # ✅ Return structure expected by Streamlit frontend
    results = [
        {
            "name": row["name"],
            "artist": row["artist"],
            "link": row["spotify_url"]
        }
        for _, row in recs_sampled.iterrows()
    ]

    return results
