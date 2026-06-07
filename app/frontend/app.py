import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# Initialy fetch inputs schemas and genres
@st.cache_data
def fetch_schema():
    response = requests.get("http://127.0.0.1:8000/openapi.json") 
    return response.json()

@st.cache_data
def fetch_genres():
    fallback_options = ["pop", "rock", "hip-hop"]
    try:
        res = requests.get(f"{API_URL}/genres")
        return res.json().get("genres", fallback_options)
    except requests.exceptions.ConnectionError:
        return fallback_options

openapi_data = fetch_schema()
available_genres = fetch_genres()

track_rules = openapi_data["components"]["schemas"]["RegressionInput"]["properties"]

# Use wide mode and collapse sidebar by default
st.set_page_config(page_title="Music App", page_icon="🎵", layout="wide")

# Helpers for creating auto validated fields
def create_auto_slider(field_name, step=None):
    rules = track_rules.get(field_name, {})
    
    title = rules.get("title", field_name.title())
    
    schema_type = rules.get("type", "number")
    
    raw_min = rules.get("minimum", 0)
    raw_max = rules.get("maximum", 1)
    raw_default = rules.get("example", "0")
    
    if schema_type == "integer":
        min_val, max_val, default_val = int(raw_min), int(raw_max), int(raw_default)
    else:
        min_val, max_val, default_val = float(raw_min), float(raw_max), float(raw_default)
        
    return st.slider(title, min_value=min_val, max_value=max_val, value=default_val, step=step)


def create_auto_number(field_name, step=None):
    rules = track_rules.get(field_name, {})
    
    title = rules.get("title", field_name.title())
    schema_type = rules.get("type", "number")
    
    raw_min = rules.get("minimum", None)
    raw_max = rules.get("maximum", None) 
    raw_default = rules.get("example", "0")
    
    if schema_type == "integer":
        min_val = int(raw_min) if raw_min is not None else None
        max_val = int(raw_max) if raw_max is not None else None
        default_val = int(raw_default)
    else:
        min_val = float(raw_min) if raw_min is not None else None
        max_val = float(raw_max) if raw_max is not None else None
        default_val = float(raw_default)
        
    return st.number_input(title, min_value=min_val, max_value=max_val, value=default_val, step=step)

# Store selected fields in dictionary
base_payload = {}

with st.sidebar:
    st.header("⚙️ Adjust Features")
    
    # General
    with st.expander("📝 General Info", expanded=True):
        track_genre = st.selectbox("Genre Context", options=available_genres)
        n_recommendations = st.number_input("Num Recommendations", min_value=1, max_value=6, value=5)
        
        base_payload["duration_ms"] = create_auto_number("duration_ms", step=1000)
        base_payload["tempo"] = create_auto_number("tempo", step=1.0)
        base_payload["key"] = create_auto_number("key", step=1)
        base_payload["time_signature"] = create_auto_number("time_signature", step=1)
        base_payload["mode"] = st.selectbox("Mode", options=[0, 1], format_func=lambda x: "Minor (0)" if x==0 else "Major (1)")
        base_payload["explicit"] = st.checkbox("Explicit", value=False)

    # Audio Properties
    with st.expander("🎛️ Audio Properties", expanded=True):
        base_payload["danceability"] = create_auto_slider("danceability")
        base_payload["energy"] = create_auto_slider("energy")
        base_payload["loudness"] = create_auto_slider("loudness", step=0.1)
        base_payload["speechiness"] = create_auto_slider("speechiness")

    # Acoustics & Emotion
    with st.expander("🎸 Acoustics & Emotion", expanded=True):
        base_payload["acousticness"] = create_auto_slider("acousticness")
        base_payload["instrumentalness"] = create_auto_slider("instrumentalness")
        base_payload["liveness"] = create_auto_slider("liveness")
        base_payload["valence"] = create_auto_slider("valence")

# Main screen
st.title("🎵 Music App")
st.markdown("Your predictions update immediately as you tweak the sidebar controls")

# Popularity and Genre side-by-side
col_pred, col_class = st.columns(2)

with col_pred:
    payload_predict = {**base_payload, "track_genre": track_genre}
    try:
        res_predict = requests.post(f"{API_URL}/predict", json=payload_predict)
        if res_predict.status_code == 200:
            pred_val = res_predict.json().get('prediction')
            st.metric(label="📈 Predicted Popularity (0-100)", value=f"{pred_val:.2f}")
        else:
            st.error("Error connecting to prediction endpoint.")
    except requests.exceptions.ConnectionError:
        st.warning("Backend disconnected.")

with col_class:
    try:
        res_classify = requests.post(f"{API_URL}/classify", json=base_payload)
        if res_classify.status_code == 200:
            genre_name = res_classify.json().get('genre_name', 'Unknown').title()
            st.metric(label="🎸 Classified Genre", value=genre_name)
        else:
            st.error("Error connecting to classification endpoint.")
    except requests.exceptions.ConnectionError:
        st.warning("Backend disconnected.")

st.divider()

# Bottom Row
st.subheader("🎧 Similar Track Recommendations")
payload_recommend = {**base_payload, "track_genre": track_genre}
try:
    res_recs = requests.post(f"{API_URL}/recommend?n_recommendations={n_recommendations}", json=payload_recommend)
    if res_recs.status_code == 200:
        recs = res_recs.json().get('recommendations', [])
        
        # Create a horizontal grid dynamically based on the number of requested recommendations
        rec_cols = st.columns(len(recs))
        
        for idx, (col, rec) in enumerate(zip(rec_cols, recs)):
            with col:
                st.info(
                    f"**{idx+1}. {rec.get('track_name')}**\n\n"
                    f"👤 {rec.get('artists')}\n\n"
                    f"🏷️ {rec.get('genre')}\n\n"
                    f"📏 Dist: {rec.get('cosine_distance'):.3f}"
                )
                
                track_id = rec.get("track_id")
                if track_id:
                    # Use standard official Spotify URL
                    spotify_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
                    
                    spotify_html = f"""
                        <iframe style="border-radius:12px; margin-top: 5px; border: none; overflow: hidden;" 
                        src="{spotify_url}" 
                        width="100%" height="160" frameBorder="0" scrolling="no" allowfullscreen="" 
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                        loading="lazy"></iframe>
                    """
                    
                    st.markdown(spotify_html, unsafe_allow_html=True)

    else:
        st.error("Error connecting to recommendation endpoint.")
except requests.exceptions.ConnectionError:
    st.warning("Backend disconnected.")