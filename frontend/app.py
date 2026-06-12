import streamlit as st
import requests
import streamlit.components.v1 as components

# --- 1. Configuration ---
st.set_page_config(page_title="Mood Mate", page_icon="🎵", layout="wide")

# This must match your FastAPI server address
API_URL = "http://localhost:8000"

# --- 2. Styling (Retaining your style + adding background) ---
def apply_custom_style():
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: 
            linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), 
            url("https://miro.medium.com/1*Gd6fni3JUnFTyNoCCcKxTw.jpeg");
            background-attachment: fixed;
            background-size: cover;
        }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

        /* Main container styling */
        .main .block-container {{
            background: rgba(14, 17, 23, 0.85);
            padding: 3rem;
            border-radius: 20px;
            margin-top: 2rem;
        }}

        /* Make all text white except main heading */
        .main .block-container * {{
            color: white !important;
        }}

        .main .block-container h1 {{
            color: #1DB954 !important;  /* main heading color */
        }}

        .song-container {{
            background: rgba(255,255,255,0.5);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #1DB954;
            transition: transform 0.2s;
            color: white !important;
        }}

        .song-container:hover {{
            transform: scale(1.02);
            background: rgba(255, 255, 255, 0.1);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            justify-content: center;
            gap: 30px;
        }}

        .confidence-text {{
            color: #1DB954 !important;
            font-weight: bold;
        }}
        </style>
    """, unsafe_allow_html=True)



apply_custom_style()

# --- 3. Helper Functions ---

def spotify_player(track_url):
    """Embeds a Spotify player using the track ID extracted from the URL."""
    if not track_url: return
    try:
        # Extract the ID and handle potential "0" prefix from recommender logic
        track_id = track_url.split('/')[-1]
        clean_id = track_id[1:] if track_id.startswith('0') else track_id
        embed_url = f"https://open.spotify.com/embed/track/{clean_id}?utm_source=generator&theme=0"
        components.iframe(embed_url, height=152, scrolling=False)
    except Exception:
        st.warning("Could not load Spotify player for this track.")

def display_mood_results(data):
    """Displays the predicted emotion and the recommended songs."""
    if not data: return
    
    if "error" in data:
        st.error(f"Backend Error: {data['error']}")
        return

    st.divider()
    
    emotion = data.get('emotion', 'Unknown').upper()
    confidence = data.get('confidence', 'N/A')
    
    st.markdown(f"### ✨ Your Mood: **{emotion}** <span class='confidence-text'>({confidence})</span>", unsafe_allow_html=True)
    
    if not data.get('songs'):
        st.info("No song recommendations found for this mood.")
        return

    # Display Songs in a grid
    cols = st.columns(2)
    for i, song in enumerate(data['songs']):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="song-container">
                <h3 style='margin:0;'>🎶 {song['name']}</h3>
                <p style='color: gray; margin:0;'>{song['artist']}</p>
            </div>
            """, unsafe_allow_html=True)
            spotify_player(song['link'])

def call_api(endpoint, files=None, data=None):
    try:
        if files:
            res = requests.post(
                f"{API_URL}{endpoint}",
                files=files
            )
        else:
            res = requests.post(
                f"{API_URL}{endpoint}",
                json=data
            )

        res.raise_for_status()
        return res.json()

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Backend. Is the FastAPI server running?")
    except Exception as e:
        st.error(f"❌ API Error: {e}")

    return None

# --- 4. Main UI ---

st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎵 Mood Mate</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>AI-Powered Music Recommendations based on your Mood</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Text & Emotions", "📸 Live Webcam", "📁 Upload Image"])

# TAB 1: Text Analysis
with tab1:
    st.write("### 🎭 Quick Select Mood")
    # Matches MobileNet-5: Angry, Happy, Sad, Surprise, Neutral
    EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'neutral']
    EMOJI_MAP = {'happy': '😊', 'sad': '😢', 'angry': '😠', 'surprise': '😲', 'neutral': '😐'}
    
    cols = st.columns(len(EMOTIONS))
    for i, emo in enumerate(EMOTIONS):
        if cols[i].button(f"{EMOJI_MAP[emo]} {emo.title()}", use_container_width=True):
            st.session_state.current_result = call_api("/predict/text", data={"text": emo})

    st.write("---")
    user_text = st.text_input("Or describe how your day is going:", placeholder="I'm feeling great today!")
    if st.button("Analyze My Feelings", type="primary"):
        if user_text:
            with st.spinner("Analyzing text..."):
                st.session_state.current_result = call_api("/predict/text", data={"text": user_text})

# TAB 2: Webcam Detection
with tab2:
    st.write("### 📸 Live Mood Detection")
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    if not st.session_state.cam_on:
        if st.button("Open Camera 📷", use_container_width=True):
            st.session_state.cam_on = True
            st.rerun()
    else:
        img_file = st.camera_input("Capture a photo of your face")
        if img_file:
            if st.button("Analyze This Photo"):
                with st.spinner("Processing Image..."):
                    files = {"file": (img_file.name, img_file.getvalue(), "image/jpeg")}
                    st.session_state.current_result = call_api("/predict/image", files=files)
        
        if st.button("Turn Off Camera"):
            st.session_state.cam_on = False
            st.rerun()

# TAB 3: File Upload
with tab3:
    st.write("### 📁 Upload a Photo")
    uploaded_file = st.file_uploader("Choose a clear photo of your face", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        st.image(uploaded_file, width=300, caption="Uploaded Image")
        if st.button("Detect Mood & Get Songs", type="primary"):
            with st.spinner("Running AI Model..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                st.session_state.current_result = call_api("/predict/image", files=files)

# --- 5. Output Area ---
if 'current_result' in st.session_state and st.session_state.current_result:
    display_mood_results(st.session_state.current_result)