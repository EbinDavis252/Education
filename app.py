import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# Load model and encoders
model = joblib.load("models/dropout_model.pkl")

# Define label mappings (same as training)
gender_map = {"Male": 1, "Female": 0, "Other": 2}
category_map = {"Gen": 0, "OBC": 2, "SC": 1, "ST": 3}
education_map = {"None": 3, "Primary": 4, "Secondary": 2, "Graduate": 0, "Postgraduate": 1}
yesno_map = {"Yes": 1, "No": 0}

# Function to preprocess data
def preprocess(df):
    df["gender"] = df["gender"].map(gender_map)
    df["category"] = df["category"].map(category_map)
    df["guardian_education"] = df["guardian_education"].map(education_map)
    df["counselling_opted"] = df["counselling_opted"].map(yesno_map)
    df["extra_curricular"] = df["extra_curricular"].map(yesno_map)
    return df

# Function to save to SQLite
def save_to_db(df):
    conn = sqlite3.connect("database/dropout.db")
    df.to_sql("student_predictions", conn, if_exists="append", index=False)
    conn.close()

# Function to suggest interventions
def suggest_intervention(row):
    interventions = []
    if row["academic_score"] < 50:
        interventions.append("Remedial classes")
    if row["attendance_rate"] < 60:
        interventions.append("Attendance monitoring")
    if row["assignments_completed"] < 50:
        interventions.append("Assignment follow-up")
    if row["counselling_opted"] == "No":
        interventions.append("Counselling support")
    return ", ".join(interventions)

# Streamlit UI
st.set_page_config("Dropout Prediction", layout="wide")
st.title("🎓 AI-Powered Student Dropout Prediction & Retention System")

# File Upload
uploaded_file = st.file_uploader("Upload Student Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    original = df.copy()  # for display
    processed_df = preprocess(df.copy())

    # Predict
    features = processed_df.drop(columns=["student_id", "name"])
    predictions = model.predict(features)
    df["predicted_dropout"] = predictions
    df["intervention"] = df.apply(suggest_intervention, axis=1)

    st.success("✅ Prediction Completed")
    st.subheader("📋 Prediction Results")
    st.dataframe(df[["student_id", "name", "academic_score", "attendance_rate", "predicted_dropout", "intervention"]])

    # Save to DB
    if st.button("📥 Save to Database"):
        save_to_db(df)
        st.success("✅ Data saved to dropout.db")

    # Download Option
    st.download_button("📄 Download Results as CSV", df.to_csv(index=False), file_name="dropout_predictions.csv")

    # Visual Summary
    st.subheader("📊 Dropout Risk Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔴 High Risk Students", df[df["predicted_dropout"] == 1].shape[0])
    with col2:
        st.metric("🟢 Low Risk Students", df[df["predicted_dropout"] == 0].shape[0])
