from transformers import pipeline

# 1. Unified mapping to match your Image Model classes:
# ['angry', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_MAPPING = {
    'joy': 'happy',
    'love': 'happy',
    'optimism': 'happy',
    'excitement': 'happy',
    'surprise': 'surprise',
    'sadness': 'sad',
    'pessimism': 'sad',
    'fear': 'sad',      # Mapped to 'sad' because image model doesn't have 'fear'
    'anger': 'angry',
    'disgust': 'angry',
    'neutral': 'neutral',
    'anticipation': 'neutral',
    'trust': 'neutral'
}

def load_text_model():
    """Loads the pre-trained BERT-Emotions-Classifier."""
    # This model is specifically trained for multi-label emotion detection
    try:
        classifier = pipeline(
            "text-classification", 
            model="ayoubkirouane/BERT-Emotions-Classifier",
            top_k=None 
        )
        return classifier
    except Exception as e:
        print(f"⚠️ Error loading BERT model: {e}")
        return None

# Load the model once globally when the server starts
TEXT_CLASSIFIER = load_text_model()

def predict_text_emotion(text: str):
    """Predicts emotion and returns a class compatible with the Music Recommender."""
    if not text.strip() or TEXT_CLASSIFIER is None:
        return "neutral"

    # 1. Get predictions from BERT
    results = TEXT_CLASSIFIER(text)[0]
    
    # 2. Sort results to find the strongest emotion
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_prediction = results[0]
    
    # 3. Confidence Threshold: If the model is unsure, default to neutral
    # Increased to 0.45 for better reliability
    if top_prediction['score'] < 0.45:
        return "neutral"

    # 4. Map the raw BERT label to one of your 5 Music Moods
    raw_label = top_prediction['label'].lower()
    final_emotion = EMOTION_MAPPING.get(raw_label, 'neutral')

    return final_emotion