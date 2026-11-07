
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path
from sklearn.metrics import pairwise_distances


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Music Taste Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROJECT PATHS AND FEATURES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

CLEANED_DATA_PATH = DATA_DIR / "spotify_tracks_cleaned.csv"
PCA_DATA_PATH = DATA_DIR / "spotify_features_pca.csv"

SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"
PCA_MODEL_PATH = MODEL_DIR / "pca_model.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 5%, #25204a 0%, transparent 35%),
        radial-gradient(circle at 90% 15%, #172e43 0%, transparent 35%),
        #0b1020;
    color: #f5f5f5;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #15152c, #101827);
    border-right: 1px solid #303653;
}

[data-testid="stSidebar"] * {
    color: #eeeeff;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #c4b5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    color: #aab3ca;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #e9e7ff;
    margin-top: 15px;
}

.metric-card {
    background: linear-gradient(135deg, #202746, #17243b);
    padding: 20px;
    border: 1px solid #354263;
    border-radius: 15px;
    margin-bottom: 10px;
}

.metric-label {
    color: #aeb9d5;
    font-size: 13px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 750;
}

.track-card {
    background: linear-gradient(135deg, #202744, #18233b);
    border: 1px solid #394567;
    border-radius: 15px;
    padding: 17px;
    margin-bottom: 12px;
}

.track-rank {
    color: #a78bfa;
    font-size: 13px;
    font-weight: bold;
}

.track-name {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    margin-top: 7px;
}

.track-artist {
    color: #b4bfd8;
    font-size: 14px;
    margin-top: 5px;
}

.track-info {
    color: #9fadd0;
    font-size: 12px;
    margin-top: 8px;
}

.hero {
    background: linear-gradient(120deg, #30245f, #163b59);
    border: 1px solid #57518b;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h2 {
    font-size: 30px;
    color: white;
}

.hero p {
    color: #d1d8ee;
}

div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 22px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #1d4ed8);
    border: none;
    color: white;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #202746, #17243b);
    padding: 16px;
    border: 1px solid #354263;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #bac4df;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stTextInput input {
    background-color: #171f35;
    color: white;
    border: 1px solid #424d70;
    border-radius: 10px;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #171f35;
    border-color: #424d70;
}

.stRadio label {
    color: #eeeeff;
}

hr {
    border-color: #343e5d;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD DATA AND MODELS
# =====================================================

@st.cache_resource
def load_recommender():

    required_files = [
        CLEANED_DATA_PATH,
        PCA_DATA_PATH,
        SCALER_PATH,
        PCA_MODEL_PATH,
        KMEANS_PATH
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    df = pd.read_csv(CLEANED_DATA_PATH)

    pca_reference = pd.read_csv(PCA_DATA_PATH)

    scaler = joblib.load(SCALER_PATH)
    pca_model = joblib.load(PCA_MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)

    X = df[FEATURES]

    X_scaled = scaler.transform(X)

    X_pca = pca_model.transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        columns=pca_reference.columns
    )

    clusters = kmeans.predict(pca_df)

    df["cluster"] = clusters

    return df, X_scaled, clusters


# =====================================================
# 5. TEXT NORMALIZATION
# =====================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"\s+", " ", value)

    value = re.sub(r"[^\w\s]", "", value)

    return value


# =====================================================
# 6. RECOMMENDATION FUNCTION
# =====================================================

def get_recommendations(
    df,
    X_scaled,
    clusters,
    selected_index,
    mode,
    result_count
):

    selected_track = df.loc[selected_index]

    selected_cluster = selected_track["cluster"]

    selected_genre = selected_track["track_genre"]

    if mode in [
        "Same genre + same cluster",
        "All genres + same cluster"
    ]:

        candidate_indices = np.where(
            clusters == selected_cluster
        )[0]

    else:

        candidate_indices = np.arange(len(df))

    candidates = df.iloc[candidate_indices].copy()

    candidates["_row_index"] = candidate_indices

    # Remove the selected song itself.

    selected_name = normalize_text(
        selected_track["track_name"]
    )

    selected_artist = normalize_text(
        selected_track["artists"]
    )

    candidates["_normalized_name"] = (
        candidates["track_name"].apply(normalize_text)
    )

    candidates["_normalized_artist"] = (
        candidates["artists"].apply(normalize_text)
    )

    candidates = candidates[
        ~(
            (candidates["_normalized_name"] == selected_name)
            &
            (candidates["_normalized_artist"] == selected_artist)
        )
    ]

    # Apply genre filtering when requested.

    if mode in [
        "Same genre + same cluster",
        "Same genre + all clusters"
    ]:

        candidates = candidates[
            candidates["track_genre"] == selected_genre
        ]

    # Remove duplicate track names and artists.

    candidates = candidates.drop_duplicates(
        subset=[
            "_normalized_name",
            "_normalized_artist"
        ],
        keep="first"
    )

    if candidates.empty:
        return candidates

    # Calculate audio-feature distance.

    candidate_positions = candidates[
        "_row_index"
    ].to_numpy()

    selected_features = X_scaled[
        selected_index
    ].reshape(1, -1)

    candidate_features = X_scaled[
        candidate_positions
    ]

    distances = pairwise_distances(
        selected_features,
        candidate_features,
        metric="euclidean"
    )[0]

    candidates["distance"] = distances

    # Rank by smallest distance.

    recommendations = candidates.sort_values(
        "distance",
        ascending=True
    ).head(result_count).copy()

    recommendations.insert(
        0,
        "Rank",
        range(1, len(recommendations) + 1)
    )

    return recommendations


# =====================================================
# 7. LOAD APPLICATION DATA
# =====================================================

try:

    with st.spinner("Preparing your music intelligence system..."):

        df, X_scaled, clusters = load_recommender()

except Exception as error:

    st.error(f"Application loading failed: {error}")
    st.stop()


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#c4b5fd;'>🎵 Music AI</h2>",
        unsafe_allow_html=True
    )

    st.caption("Music Taste Intelligence System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🎧 Discover Music",
            "📊 Analytics",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.success("Models loaded successfully")

    st.caption("PCA + K-Means + Euclidean Distance")

    st.divider()

    st.caption("Unsupervised Machine Learning Project")


# =====================================================
# 9. DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":

    st.markdown(
        "<div class='main-title'>Music Taste Intelligence</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Discover patterns in music through "
        "unsupervised machine learning.</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='hero'>
        <h2>🎶 Discover Your Next Favourite Track</h2>
        <p>Explore music collections, discover audio similarities,
        and understand how machine learning groups songs.</p>
    </div>
    """, unsafe_allow_html=True)

    # Main metrics.

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Tracks",
        f"{len(df):,}"
    )

    col2.metric(
        "Music Genres",
        f"{df['track_genre'].nunique():,}"
    )

    col3.metric(
        "Clusters",
        len(np.unique(clusters))
    )

    col4.metric(
        "Audio Features",
        len(FEATURES)
    )

    st.divider()

    st.markdown(
        "<div class='section-title'>📈 Cluster Distribution</div>",
        unsafe_allow_html=True
    )

    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts,
        color="#8b5cf6"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🎼 Top Genres")

        genre_counts = df[
            "track_genre"
        ].value_counts().head(10)

        st.bar_chart(
            genre_counts,
            color="#06b6d4"
        )

    with col2:

        st.markdown("### 🎧 Cluster Overview")

        cluster_table = (
            df["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Tracks")
        )

        st.dataframe(
            cluster_table,
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# 10. DISCOVER MUSIC PAGE
# =====================================================

elif page == "🎧 Discover Music":

    st.markdown(
        "<div class='main-title'>Discover Music</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Find tracks with similar audio "
        "characteristics.</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-title'>🔍 Search Your Favourite Track</div>",
        unsafe_allow_html=True
    )

    search_query = st.text_input(
        "Song name",
        placeholder="Try Shape of You, Believer, Perfect..."
    )

    if search_query.strip():

        exact_matches = df[
            df["track_name"].str.strip().str.lower()
            == search_query.strip().lower()
        ]

        if not exact_matches.empty:

            matches = exact_matches

        else:

            matches = df[
                df["track_name"].str.contains(
                    search_query.strip(),
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        if matches.empty:

            st.warning("No matching tracks found.")

        else:

            st.success(
                f"{len(matches)} matching track(s) found."
            )

            def track_label(index):

                track = df.loc[index]

                return (
                    f"{track['track_name']} | "
                    f"{track['artists']} | "
                    f"{track['track_genre']}"
                )

            selected_index = st.selectbox(
                "Choose your track",
                options=matches.index.tolist(),
                format_func=track_label
            )

            selected_track = df.loc[selected_index]

            st.divider()

            st.markdown("### 🎵 Selected Track")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Track",
                selected_track["track_name"]
            )

            col2.metric(
                "Artist",
                selected_track["artists"]
            )

            col3.metric(
                "Cluster",
                int(selected_track["cluster"])
            )

            st.write(
                f"**Genre:** {selected_track['track_genre']}"
            )

            st.write(
                f"**Popularity:** {selected_track['popularity']}"
            )

            st.divider()

            st.markdown("### ⚙️ Recommendation Settings")

            col1, col2 = st.columns(2)

            with col1:

                recommendation_mode = st.selectbox(
                    "Recommendation type",
                    [
                        "Same genre + same cluster",
                        "All genres + same cluster",
                        "Same genre + all clusters",
                        "All genres + all clusters"
                    ]
                )

            with col2:

                result_count = st.slider(
                    "Number of recommendations",
                    min_value=5,
                    max_value=20,
                    value=10
                )

            if st.button(
                "✨ Generate Music Recommendations",
                use_container_width=True
            ):

                recommendations = get_recommendations(
                    df,
                    X_scaled,
                    clusters,
                    selected_index,
                    recommendation_mode,
                    result_count
                )

                if recommendations.empty:

                    st.warning(
                        "No recommendations found. "
                        "Try a different recommendation type."
                    )

                else:

                    st.session_state["recommendations"] = (
                        recommendations
                    )

                    st.session_state["recommendation_track"] = (
                        selected_track["track_name"]
                    )

                    st.session_state["recommendation_mode"] = (
                        recommendation_mode
                    )

            # Display results from session state.

            if "recommendations" in st.session_state:

                recommendations = st.session_state[
                    "recommendations"
                ]

                if (
                    st.session_state.get("recommendation_track")
                    == selected_track["track_name"]
                ):

                    st.divider()

                    st.markdown(
                        "<div class='section-title'>"
                        "🌟 Recommended For You</div>",
                        unsafe_allow_html=True
                    )

                    st.caption(
                        "Mode: "
                        + st.session_state["recommendation_mode"]
                    )

                    for _, track in recommendations.iterrows():

                        st.markdown(
                            f"""
                            <div class="track-card">
                                <div class="track-rank">
                                    RANK #{int(track['Rank'])}
                                </div>
                                <div class="track-name">
                                    🎵 {track['track_name']}
                                </div>
                                <div class="track-artist">
                                    {track['artists']}
                                </div>
                                <div class="track-info">
                                    Genre: {track['track_genre']}
                                    &nbsp; | &nbsp;
                                    Cluster: {int(track['cluster'])}
                                    &nbsp; | &nbsp;
                                    Distance: {track['distance']:.4f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.caption(
                        "Lower distance means closer audio-feature "
                        "similarity. It is not a probability."
                    )

                    csv_data = recommendations[
                        [
                            "Rank",
                            "track_name",
                            "artists",
                            "track_genre",
                            "cluster",
                            "distance"
                        ]
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Recommendations",
                        data=csv_data,
                        file_name="music_recommendations.csv",
                        mime="text/csv"
                    )


# =====================================================
# 11. ANALYTICS PAGE
# =====================================================

elif page == "📊 Analytics":

    st.markdown(
        "<div class='main-title'>Music Analytics</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Explore the characteristics "
        "of your music dataset and discovered clusters.</div>",
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Cluster Distribution")

    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts,
        color="#a78bfa"
    )

    st.markdown("### 🎼 Genre Distribution")

    genre_counts = (
        df["track_genre"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(
        genre_counts,
        color="#22d3ee"
    )

    st.divider()

    st.markdown("### 🎚️ Audio Feature Analysis")

    selected_features = st.multiselect(
        "Select audio features",
        options=FEATURES,
        default=[
            "danceability",
            "energy",
            "acousticness",
            "valence"
        ]
    )

    if selected_features:

        cluster_profiles = (
            df.groupby("cluster")[selected_features]
            .mean()
            .round(3)
        )

        st.dataframe(
            cluster_profiles,
            use_container_width=True
        )

        st.markdown("### Feature Comparison by Cluster")

        selected_feature = st.selectbox(
            "Choose a feature to visualize",
            selected_features
        )

        feature_chart = (
            df.groupby("cluster")[selected_feature]
            .mean()
        )

        st.bar_chart(
            feature_chart,
            color="#c084fc"
        )

    st.divider()

    st.markdown("### 📋 Cluster Statistics")

    cluster_summary = (
        df.groupby("cluster")
        .agg(
            Track_Count=("track_name", "count"),
            Average_Popularity=("popularity", "mean"),
            Average_Energy=("energy", "mean"),
            Average_Danceability=("danceability", "mean"),
            Average_Valence=("valence", "mean")
        )
        .round(3)
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )


# =====================================================
# 12. ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ️ About Project":

    st.markdown(
        "<div class='main-title'>About This Project</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Understanding music through "
        "unsupervised machine learning.</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='hero'>
        <h2>🎵 Music Taste Intelligence & Clustering System</h2>
        <p>A machine learning project that discovers hidden
        patterns in music using audio characteristics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objective")

    st.write(
        "The objective is to group music tracks based on their "
        "audio features and recommend tracks with similar "
        "characteristics."
    )

    st.markdown("### 🧠 Machine Learning Algorithms")

    algorithms = pd.DataFrame({
        "Algorithm": [
            "K-Means Clustering",
            "DBSCAN",
            "Hierarchical Clustering",
            "Principal Component Analysis"
        ],
        "Purpose": [
            "Groups tracks into 7 clusters",
            "Identifies dense groups and noise",
            "Explores hierarchical relationships",
            "Reduces 10 features to 8 components"
        ]
    })

    st.dataframe(
        algorithms,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🎚️ Audio Features Used")

    st.write(", ".join(FEATURES))

    st.markdown("### 🔄 Project Workflow")

    st.markdown("""
    1. Dataset exploration
    2. Data cleaning
    3. Feature selection
    4. Feature scaling
    5. PCA dimensionality reduction
    6. K-Means clustering
    7. DBSCAN clustering
    8. Hierarchical clustering
    9. Cluster evaluation and comparison
    10. Music recommendation engine
    11. Streamlit deployment
    """)

    st.markdown("### 📁 Project Dataset")

    st.write(
        f"Total records: {len(df):,}"
    )

    st.write(
        f"Total genres: {df['track_genre'].nunique()}"
    )

    st.write(
        f"Audio features: {len(FEATURES)}"
    )

    st.info(
        "This dataset contains track-level Spotify audio features. "
        "It does not contain individual users' listening histories. "
        "Recommendations are based on audio-feature similarity."
    )


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8e9ab8; padding:15px;">
        🎵 Music Taste Intelligence
        <br>
        Unsupervised Machine Learning Project
        <br>
        Built with Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
