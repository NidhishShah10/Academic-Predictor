#Main file for the Streamlit app

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Academic Predictor",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "academic_predictor_model.pkl"
FEATURES_PATH = BASE_DIR / "outputs" / "feature_columns.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error(
        "Saved model files not found. "
        "Run python academic_predictor.py first."
    )
    st.stop()

model = load_model()
feature_columns = joblib.load(FEATURES_PATH)

st.title("🎓 Academic Predictor")
st.write(
    "Predict whether a student is academically At Risk or "
    "Satisfactory using a trained machine learning model."
)

st.info(
    "This application is an educational demonstration. "
    "Predictions are not official academic assessments."
)

st.divider()

with st.form("student_form"):
    st.subheader("📚 Academic Information")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.number_input(
            "Hours Studied (per week)",
            min_value=0,
            max_value=100,
            value=20
        )
        attendance = st.slider(
            "Attendance (%)", 0, 100, 85
        )
        previous_scores = st.slider(
            "Previous Exam Score", 0, 100, 70
        )
        tutoring_sessions = st.number_input(
            "Tutoring Sessions",
            min_value=0,
            max_value=30,
            value=1
        )
        sleep_hours = st.slider(
            "Sleep Hours (per night)", 0, 15, 7
        )

    with col2:
        motivation = st.selectbox(
            "Motivation Level",
            ["Low", "Medium", "High"],
            index=1
        )
        parental_involvement = st.selectbox(
            "Parental Involvement",
            ["Low", "Medium", "High"],
            index=1
        )
        access_resources = st.selectbox(
            "Access to Resources",
            ["Low", "Medium", "High"],
            index=1
        )
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["Yes", "No"]
        )
        physical_activity = st.slider(
            "Physical Activity (hours per week)", 0, 20, 3
        )

    st.divider()
    st.subheader("🏠 Personal and Environmental Information")

    col3, col4 = st.columns(2)

    with col3:
        family_income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"],
            index=1
        )
        parental_education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"]
        )
        internet_access = st.selectbox(
            "Internet Access",
            ["Yes", "No"]
        )
        school_type = st.selectbox(
            "School Type",
            ["Public", "Private"]
        )
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col4:
        teacher_quality = st.selectbox(
            "Teacher Quality",
            ["Low", "Medium", "High"],
            index=1
        )
        peer_influence = st.selectbox(
            "Peer Influence",
            ["Negative", "Neutral", "Positive"],
            index=1
        )
        distance_home = st.selectbox(
            "Distance from Home",
            ["Near", "Moderate", "Far"]
        )
        learning_disabilities = st.selectbox(
            "Learning Disabilities",
            ["No", "Yes"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Academic Risk",
        use_container_width=True
    )

if submitted:
    student_data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_home,
        "Gender": gender
    }

    student_df = pd.DataFrame([student_data])
    student_df = student_df[feature_columns]

    prediction = int(model.predict(student_df)[0])
    risk_index = list(model.classes_).index(1)
    risk_probability = float(
        model.predict_proba(student_df)[0][risk_index]
    )

    st.divider()
    st.subheader("📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ Predicted Result: AT RISK")
        else:
            st.success("✅ Predicted Result: SATISFACTORY")

    with result_col2:
        st.metric(
            "Estimated Probability of At Risk",
            f"{risk_probability * 100:.2f}%"
        )

    st.progress(min(max(risk_probability, 0.0), 1.0))

    st.caption(
        "This probability is a model estimate, not a "
        "validated real-world probability of academic failure."
    )

    if prediction == 1:
        st.info(
            "Consider reviewing study habits, attendance, "
            "and available academic support resources."
        )
    else:
        st.info(
            "Continue maintaining productive study habits "
            "and monitoring academic progress."
        )