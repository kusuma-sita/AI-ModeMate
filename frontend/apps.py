import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# Config & Constants
# =========================
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Emotion → Music Recommender",
    page_icon="🎧",
    layout="wide"
)

# =========================
# Global Styling
# =========================
APP_CSS = """
<style>
/* Background gradient */
.stApp {
  background: linear-gradient(135deg, #0d0b28 0%, #2c247d 50%, #0e0a3c 100%);
  color: #E9E9EF !important;
  font-family: 'Inter', sans-serif;
}

/* Hide default header & footer */
header, footer {visibility: hidden;}

/* Title Card */
.title-card {
  background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
  border-radius: 20px;
  padding: 1.3rem 1.6rem;
  box-shadow: 0 6px 25px rgba(0,0,0,0.3);
  margin-bottom: 1.2rem;
  text-align: center;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  color: #FFFFFF !important;
  letter-spacing: 0.3px;
}

/* Buttons */
.stButton>button {
  background: linear-gradient(135deg, #6e3cf7 0%, #7b5cfb 100%);
  color: white;
  border-radius: 12px;
  border: none;
  font-weight: 600;
  padding: 0.6rem 1.1rem;
  transition: all 0.3s ease;
}
.stButton>button:hover {
  background: linear-gradient(135deg, #a98ffb 0%, #8b68fc 100%);
  transform: scale(1.03);
}

/* Toggle switch */
label[data-baseweb="toggle"] {
  color: #ddd;
  font-weight: 600;
}

/* Info cards */
.music-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 1rem 1.3rem;
  margin-top: 0.5rem;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* Song card */
.song {
  background: rgba(255,255,255,0.07);
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  transition: all 0.2s ease;
}
.song:hover {
  background: rgba(255,255,255,0.12);
  transform: scale(1.02);
}
a.song-link {
  color: #a3e8ff !important;
  text-decoration: none;
}
a.song-link:hover {
  text-decoration: underline;
}

/* Divider */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  margin: 1rem 0 1.3rem 0;
}

/* Tabs */
[data-baseweb="tab"] {
  color: #eee;
  font-weight: 600;
}

/* Expander */
.streamlit-expanderHeader {
  font-weight: 700;
  color: #fff !important;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# =========================
# Header Section
# =========================
st.markdown(
    """
    <div class='title-card'>
        <h1>🎧 Emotion → Music Recommender</h1>
        <p style='color:#ddd; font-size:1.05rem;'>
        Detect your emotion — from text, a photo, or webcam — and get a personalized playlist 
        that <strong>matches or uplifts</strong> your mood.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# =========================
# Session State
# =========================
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []
if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = False

# =========================
# Helper Functions
# =========================
def post_text_emotion(text: str):
    r = requests.post(f"{FASTAPI_URL}/predict/text", json={"text": text})
    r.raise_for_status()
    j = r.json()
    return j.get("mapped_emotion") or j.get("emotion") or j.get("detected_text_emotion")

def post_face_emotion_from_bytes(image_bytes: bytes):
    files = {"file": ("face.jpg", image_bytes, "image/jpeg")}
    r = requests.post(f"{FASTAPI_URL}/predict/face-image", files=files)
    r.raise_for_status()
    return r.json().get("emotion")

def post_webcam_emotion():
    r = requests.get(f"{FASTAPI_URL}/predict/face-webcam")
    r.raise_for_status()
    return r.json().get("emotion")

def post_recommendations(emotion: str, uplift: bool):
    payload = {"emotion": emotion, "uplift": uplift}
    r = requests.post(f"{FASTAPI_URL}/recommend", json=payload)
    r.raise_for_status()
    return r.json()

def render_recommendations(recs):
    if not recs:
        st.info("No recommendations yet.")
        return
    for item in recs:
        name = item.get("name", "Unknown Track")
        artist = item.get("artist", "Unknown Artist")
        link = item.get("link", "#")
        st.markdown(
            f"""
            <div class="song">
                🎵 <strong>{name}</strong> — <em>{artist}</em><br>
                <a class="song-link" href="{link}" target="_blank">▶ Open on Spotify</a>
            </div>
            """, unsafe_allow_html=True
        )

# =========================
# Tabs
# =========================
tab_face, tab_text, tab_logic = st.tabs([
    "🖼️ Face (Upload / Camera)",
    "💬 Text",
    "📊 Recommender Logic"
])

# -------------------------
# 📊 Recommender Logic (Compact & Professional)
# -------------------------
with tab_logic:
    st.markdown("## 🎼 How the Emotion → Music Recommender Works")

    st.markdown(
        """
        <div class='music-card'>
        The system maps emotions onto a **Valence–Arousal (Energy)** space rather than simple labels.
        Songs are matched or uplifted based on their <strong>valence</strong> (pleasantness) and
        <strong>energy</strong> (arousal) values — features extracted from sources such as Spotify or the MUSE dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========== 1️⃣ COMPACT VALENCE–AROUSAL MAP + DESCRIPTION ==========
    col_plot, col_text = st.columns([1, 2], vertical_alignment="center")

    with col_plot:
        fig, ax = plt.subplots(figsize=(3.5, 3.3))
        ax.axhline(0.5, color='white', lw=1)
        ax.axvline(0.5, color='white', lw=1)

        # quadrants background
        ax.fill_betweenx([0.5, 1], 0.5, 1, color="#3CB371", alpha=0.25)   # Happy
        ax.fill_betweenx([0, 0.5], 0.5, 1, color="#FFD700", alpha=0.25)   # Relaxed
        ax.fill_betweenx([0.5, 1], 0, 0.5, color="#FF6B6B", alpha=0.25)   # Angry
        ax.fill_betweenx([0, 0.5], 0, 0.5, color="#6495ED", alpha=0.25)   # Sad

        # emotion points
        emotions = {
            "😡 Angry": (0.25, 0.85),
            "😃 Happy": (0.8, 0.75),
            "😢 Sad": (0.25, 0.25),
            "🙂 Relaxed": (0.75, 0.35),
            "😐 Neutral": (0.5, 0.5)
        }
        for label, (x, y) in emotions.items():
            ax.text(x, y, label, color='white', ha='center', va='center', fontsize=9, weight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Valence → Pleasantness", fontsize=9, color='white')
        ax.set_ylabel("Energy / Arousal ↑", fontsize=9, color='white')
        ax.set_facecolor("#1b153a")
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig, transparent=True)

    with col_text:
        st.markdown(
            """
            **Valence–Arousal Model**

            - **Valence** → how *pleasant* or *positive* the emotion feels  
            - **Arousal (Energy)** → how *activated* or *intense* the emotion is  

            Emotions are positioned in this 2-D space:
            - 😃 **Happy** → high valence, high arousal  
            - 😡 **Angry** → low valence, high arousal  
            - 😢 **Sad** → low valence, low arousal  
            - 🙂 **Relaxed** → high valence, low arousal  
            - 😐 **Neutral** → mid valence & energy  
            """
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== 2️⃣ CONGRUENT TABLE ==========
    st.markdown("### 🎯 Mood-Congruent Recommendation Logic")

    congruent = pd.DataFrame({
        "Emotion": ["Sad 😢", "Happy 😃", "Angry 😡", "Surprised 😲", "Neutral 😐"],
        "Valence": ["Low (<0.4)", "High (>0.6)", "Low (<0.4)", "Mid (0.4–0.7)", "Mid (0.4–0.6)"],
        "Energy": ["Low (<0.5)", "High (>0.5)", "High (>0.7)", "High (>0.6)", "Mid (0.4–0.6)"],
        "Psychological Basis": [
            "Sadness = unpleasant + calm (slow tempo, minor key)",
            "Happiness = pleasant + energetic (major key, upbeat rhythm)",
            "Anger = unpleasant + activated (fast, loud, aggressive tones)",
            "Surprise = mixed valence but high arousal (sudden shifts)",
            "Neutral = balanced affect (steady, smooth rhythm)"
        ]
    })
    st.dataframe(congruent, hide_index=True, width='stretch')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== 3️⃣ UPLIFTING TABLE ==========
    st.markdown("### 🌤️ Mood-Uplifting Recommendation Logic")

    uplifting = pd.DataFrame({
        "Detected Emotion": ["Sad 😢", "Angry 😡", "Fearful 😱", "Happy 😃", "Neutral 😐"],
        "Target Valence": ["↑ (0.6–0.9)", "↑ (0.5–0.8)", "↑ (0.6–0.9)", "Maintain (0.6–1.0)", "↔ (0.4–0.6)"],
        "Target Energy": ["↑ / ↔ (0.4–0.7)", "↓ (0.3–0.6)", "↓ / ↔ (0.3–0.6)", "Maintain (0.5–1.0)", "↔ (0.4–0.6)"],
        "Psychological Purpose": [
            "Gently elevate mood with warm, hopeful tracks",
            "De-escalate tension with calm tones",
            "Provide comfort & reassurance",
            "Sustain positivity and engagement",
            "Encourage balanced exploration"
        ]
    })
    st.dataframe(uplifting, hide_index=True, width='stretch')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ========== 4️⃣ TECH SUMMARY ==========
    st.markdown(
        """
        <div class='music-card'>
        <h4>🔬 Technical Summary</h4>
        <p>
        The recommender converts your detected emotion into approximate <code>(valence, energy)</code> coordinates 
        and selects songs from the dataset that match these regions:
        </p>
        <ul>
            <li>🎯 <strong>Congruent mode</strong> → reinforces your current state (mood validation)</li>
            <li>🌤️ <strong>Uplifting mode</strong> → shifts toward higher valence or balanced energy (mood regulation)</li>
        </ul>
        <p>
        Thresholds are derived from affective computing literature and Spotify’s musical emotion metrics.  
        The approach balances empathy and regulation — validating emotions while promoting well-being.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------
# Tab: Face (Upload / Camera using Streamlit)
# -------------------------
with tab_face:
    st.markdown("### 🖼️ Detect from Image or Camera")
    st.markdown(
        "<div class='music-card'>Choose how you'd like to capture your emotion — upload an image or open your camera. "
        "We'll detect your mood and spin a playlist that fits your vibe. 🎶</div>",
        unsafe_allow_html=True
    )

    # --- Capture Mode Selector ---
    st.markdown("<br>", unsafe_allow_html=True)
    capture_mode = st.radio(
        "Select Mode:",
        ["📁 Upload Image", "🎥 Use Camera"],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

    camera_image = None
    uploaded = None

    # --- Upload Mode ---
    if capture_mode == "📁 Upload Image":
        uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

    # --- Camera Mode ---
    else:
        if not st.session_state.camera_enabled:
            if st.button("🎥 Enable Camera"):
                st.session_state.camera_enabled = True
                st.rerun()
        else:
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                st.image(camera_image, caption="Captured from camera", use_container_width=True)
            if st.button("❌ Close Camera"):
                st.session_state.camera_enabled = False
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    uplift = st.toggle("🎶 Cheer me up instead of matching my mood", value=False)

    # --- Analyze and Recommend Button ---
    if st.button("✨ Analyze & Recommend (Face)"):
        try:
            img_bytes = None
            if uploaded is not None:
                img_bytes = uploaded.read()
            elif st.session_state.get("camera_enabled") and "camera_image" in locals() and camera_image is not None:
                img_bytes = camera_image.getvalue()

            if not img_bytes:
                st.warning("Please upload or capture an image first.")
            else:
                emo = post_face_emotion_from_bytes(img_bytes)
                st.session_state.last_emotion = emo
                st.success(f"Detected emotion: **{emo}**")

                recs = post_recommendations(emo, uplift)
                st.session_state.last_recs = recs

                st.markdown("### 🎧 Your Playlist")
                render_recommendations(recs)
        except requests.exceptions.ConnectionError:
            st.error("Could not reach FastAPI. Is it running?")
        except requests.HTTPError as e:
            st.error(f"Server error: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")



# -------------------------
# Webcam (Backend) Tab
# -------------------------


# -------------------------
# Text Tab
# -------------------------
with tab_text:
    st.markdown("<h3>💬 Detect from Text</h3>", unsafe_allow_html=True)
    st.markdown("<div class='music-card'>Write how you feel — your emotion is detected by the BERT model and mapped to a mood for music recommendations.</div>", unsafe_allow_html=True)
    
    user_text = st.text_area("How are you feeling today?", "I'm feeling great and energized!")
    uplift3 = st.toggle("🎶 Cheer me up instead of matching my mood", key="uplift3", value=False)

    if st.button("✨ Analyze & Recommend (Text)"):
        try:
            emo = post_text_emotion(user_text)
            st.session_state.last_emotion = emo
            st.success(f"Detected emotion: **{emo}**")

            recs = post_recommendations(emo, uplift3)
            st.session_state.last_recs = recs

            st.markdown("### 🎧 Your Playlist")
            render_recommendations(recs)
        except Exception as e:
            st.error(f"⚠️ {e}")


# =========================
# Footer / Last Results
# =========================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
with st.expander("🕒 Last Detected Emotion & Playlist"):
    st.write("**Emotion:**", st.session_state.last_emotion or "None yet")
    render_recommendations(st.session_state.last_recs)
