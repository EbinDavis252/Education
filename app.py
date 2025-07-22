import streamlit as st
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Page config
st.set_page_config(page_title="Student Dropout Prediction System", layout="wide")
st.title("🎓 AI-Powered Student Dropout Prediction & Retention System")

# Sidebar menu
menu = ["Upload Data & Predict", "View Predictions", "Intervention Suggestions"]
choice = st.sidebar.selectbox("Menu", menu)

# DB connection
def get_connection():
    return sqlite3.connect("dropout.db")

# Train model inside app
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("data/students_train.csv")
    except FileNotFoundError:
        st.error("Training data not found. Please upload students_train.csv in /data/")
        return None

    label_encoders = {}
    for col in ["gender", "category", "guardian_education", "counselling_opted", "extra_curricular"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=["student_id", "name", "dropout"], errors="ignore")
    y = df["dropout"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, label_encoders

model_data = train_model()
if model_data is None:
    st.stop()

model, label_encoders = model_data

# Upload & Predict
if choice == "Upload Data & Predict":
    st.subheader("📂 Upload Student Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()

        # Apply encoders
        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])

        X = df.drop(columns=["student_id", "name"], errors="ignore")
        predictions = model.predict(X)
        original_df["predicted_dropout"] = predictions

        st.success("✅ Predictions completed!")
        st.dataframe(original_df)

        # Save to DB
        conn = get_connection()
        cursor = conn.cursor()

        for _, row in original_df.iterrows():
            cursor.execute("""
                INSERT INTO student_predictions (
                    student_id, name, age, gender, category, family_income,
                    guardian_education, attendance_rate, academic_score,
                    assignments_completed, library_usage, hours_spent_online,
                    counselling_opted, extra_curricular, predicted_dropout
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(row[col] for col in [
                "student_id", "name", "age", "gender", "category", "family_income",
                "guardian_education", "attendance_rate", "academic_score",
                "assignments_completed", "library_usage", "hours_spent_online",
                "counselling_opted", "extra_curricular", "predicted_dropout"]))

        conn.commit()
        conn.close()
        st.success("📥 Predictions saved to database!")

# View Saved Predictions
elif choice == "View Predictions":
    st.subheader("📊 View Saved Predictions")
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM student_predictions", conn)
    conn.close()

    st.dataframe(df)

# View Interventions
elif choice == "Intervention Suggestions":
    st.subheader("🧬 Recommended Interventions for At-Risk Students")
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM student_predictions WHERE predicted_dropout = 1", conn)
    conn.close()

    if df.empty:
        st.info("✅ No at-risk students found.")
    else:
        for _, row in df.iterrows():
            st.markdown(f"**{row['name']}** (ID: {row['student_id']})")
            st.markdown("- Low academic performance — recommend academic counselling")
            st.markdown("- Offer peer mentoring or tutoring")
            st.markdown("- Involve parents/guardians")
            st.markdown("- Schedule regular follow-ups\n")
