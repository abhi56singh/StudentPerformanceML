import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
with open("student_performance_model.pkl", "rb") as file:
    model = pickle.load(file)
# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []
# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction System")

st.write(
    "Predict a student's final grade using a Machine Learning model."
)

st.divider()

# -----------------------------
# Student Information
# -----------------------------
st.subheader("📚 Student Information")

col1, col2 = st.columns(2)

with col1:
    studytime = st.number_input(
        "Study Time (1-4)",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.number_input(
        "Past Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=93,
        value=5
    )

with col2:
    g1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    g2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Final Grade", use_container_width=True):

    input_data = pd.DataFrame({
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G1": [g1],
        "G2": [g2]
    })

    prediction = model.predict(input_data)[0]

    # Keep prediction within 0-20
    prediction = max(0, min(20, prediction))
    
    # Save prediction to history
    st.session_state.history.append({
    "Study Time": studytime,
    "Failures": failures,
    "Absences": absences,
    "G1": g1,
    "G2": g2,
    "Predicted G3": round(prediction, 2)
})

    st.subheader("📊 Prediction Result")

    st.success(
        f"Predicted Final Grade: {prediction:.2f} / 20"
    )

    # Grade interpretation
    if prediction >= 16:
        st.info("🌟 Excellent predicted performance")
    elif prediction >= 12:
        st.info("👍 Good predicted performance")
    elif prediction >= 10:
        st.warning("📖 Moderate predicted performance")
    else:
        st.error("⚠️ Student may need additional academic support")

st.divider()

# -----------------------------
# Model Information
# -----------------------------
st.subheader("🤖 Model Information")

st.write("""
**Algorithm:** Random Forest Regression

**Input Features:** Study Time, Past Failures, Absences, G1, G2

**Target:** Final Grade (G3)

**R² Score:** 0.873

**Mean Absolute Error:** 1.053
""")
# -----------------------------
# Prediction History
# -----------------------------
if st.session_state.history:

    st.divider()

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )