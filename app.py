
import pickle
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Load the trained model + scaler + exact feature order
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("breast_cancer.pkl", "rb") as file:
        model_data = pickle.load(file)
    return model_data["model"], model_data["scaler"], model_data["features"]

model, scaler, features = load_model()

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #f7f9fc;
    }

    .hero {
        background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
        padding: 32px 38px;
        border-radius: 22px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.18);
    }

    .hero h1 {
        font-size: 36px;
        margin: 0 0 8px 0;
        font-weight: 750;
        text-align: center;
    }

    .hero p {
        font-size: 16px;
        margin: 0;
        opacity: 0.94;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #202438;
        margin: 22px 0 6px 0;
    }

    .section-subtitle {
        color: #687083;
        font-size: 14px;
        margin-bottom: 16px;
    }

    .info-card {
        background: white;
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid #e8eaf0;
        margin-bottom: 18px;
    }

    .result-benign {
        background: #ecfdf3;
        border: 1px solid #a7f3c6;
        color: #166534;
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        margin-top: 22px;
    }

    .result-malignant {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #9f1239;
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        margin-top: 22px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .result-note {
        font-size: 14px;
        opacity: 0.9;
    }

    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        padding: 14px 17px;
        border-radius: 12px;
        color: #9a3412;
        font-size: 13px;
        margin-top: 18px;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 22px 25px;
        border-radius: 18px;
        border: 1px solid #e8eaf0;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 700;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🩺 AI BASED BREAST CANCER PREDICTOR</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input defaults taken from the project's prediction example.
# Users can replace every value.
# ---------------------------------------------------------
defaults = {
    "texture_mean": 14.36,
    "smoothness_mean": 0.09779,
    "compactness_mean": 0.08129,
    "concave_points_mean": 0.04781,
    "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766,
    "texture_se": 0.7886,
    "area_se": 23.56,
    "smoothness_se": 0.008462,
    "compactness_se": 0.01460,
    "concavity_se": 0.02387,
    "concave_points_se": 0.01315,
    "symmetry_se": 0.01980,
    "fractal_dimension_se": 0.002300,
    "texture_worst": 19.26,
    "area_worst": 711.2,
    "smoothness_worst": 0.14400,
    "compactness_worst": 0.17730,
    "concavity_worst": 0.2390,
    "concave_points_worst": 0.1288,
    "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259,
}

# Labels are intentionally human-readable while the actual dataframe columns
# remain exactly identical to the trained model's feature names.
labels = {
    "texture_mean": "Texture Mean",
    "smoothness_mean": "Smoothness Mean",
    "compactness_mean": "Compactness Mean",
    "concave_points_mean": "Concave Points Mean",
    "symmetry_mean": "Symmetry Mean",
    "fractal_dimension_mean": "Fractal Dimension Mean",
    "texture_se": "Texture SE",
    "area_se": "Area SE",
    "smoothness_se": "Smoothness SE",
    "compactness_se": "Compactness SE",
    "concavity_se": "Concavity SE",
    "concave_points_se": "Concave Points SE",
    "symmetry_se": "Symmetry SE",
    "fractal_dimension_se": "Fractal Dimension SE",
    "texture_worst": "Texture Worst",
    "area_worst": "Area Worst",
    "smoothness_worst": "Smoothness Worst",
    "compactness_worst": "Compactness Worst",
    "concavity_worst": "Concavity Worst",
    "concave_points_worst": "Concave Points Worst",
    "symmetry_worst": "Symmetry Worst",
    "fractal_dimension_worst": "Fractal Dimension Worst",
}

groups = {
    "Mean Features": features[:6],
    "Standard Error (SE) Features": features[6:14],
    "Worst Features": features[14:],
}

values = {}

with st.form("prediction_form"):
    for group_name, group_features in groups.items():
        st.markdown(f'<div class="section-title">{group_name}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Enter the numerical measurement for each feature.</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(3)
        for idx, feature in enumerate(group_features):
            with cols[idx % 3]:
                values[feature] = st.number_input(
                    labels[feature],
                    value=float(defaults[feature]),
                    format="%.6f",
                    help=f"Model feature: {feature}",
                    key=feature,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Diagnosis", use_container_width=True)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    input_data = {feature: values[feature] for feature in features}
    input_df = pd.DataFrame([input_data])

    # Match exact training order before scaling.
    input_df = input_df[features]

    # Use the exact scaler saved with the model.
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    # SVC was trained with probability=True in the notebook.
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_scaled)[0][int(prediction)])

    if int(prediction) == 1:
        st.markdown("""
        <div class="result-malignant">
            <div class="result-title">⚠️ Predicted: Malignant</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-benign">
            <div class="result-title">✓ Predicted: Benign</div>
        </div>
        """, unsafe_allow_html=True)

    if probability is not None:
        st.metric("Model probability for predicted class", f"{probability * 100:.2f}%")

# ---------------------------------------------------------
# Project information
# ---------------------------------------------------------
with st.expander("Model / project details"):
    st.write("**Model:** Support Vector Classifier (SVC)")
    st.write("**Input features:** 22")
    st.write("**Preprocessing:** StandardScaler saved inside `breast_cancer.pkl`")
    st.write("**Class mapping from the project:** 0 = Benign, 1 = Malignant")
    st.write("**Feature order:**")
    st.code("\n".join(features))
