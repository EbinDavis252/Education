import streamlit as st
import pandas as pd
import os
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# Create data directory if not exists
os.makedirs("data", exist_ok=True)
os.makedirs("database", exist_ok=True)

st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select Option", ["Upload Data & Predict"])

st.markdown("<h1 style='text-align: center;'>🎓 AI-Powered Student Dropout Prediction & Retention System</h1>", unsafe_allow_html=True)

if menu == "Upload Data & Predict":
    uploaded_file = st.file_uploader("📤 Upload Training Dataset (CSV)", type=["csv"])

    if uploaded_file:
        file_path = os.path.join("data", "students_train.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ File uploaded successfully!")

    # Proceed if file exists
    if os.path.exists("data/students_train.csv"):
        df = pd.read_csv("data/students_train.csv")

        st.subheader("📊 Preview of Uploaded Dataset")
        st.dataframe(df.head(10))

        if "Dropout" not in df.columns:
            st.error("❌ 'Dropout' column not found in data. Please include target variable.")
        else:
            # Encode target and categorical
            df = df.dropna()
            le = LabelEncoder()
            for col in df.select_dtypes(include="object"):
                df[col] = le.fit_transform(df[col])

            X = df.drop("Dropout", axis=1)
            y = df["Dropout"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📈 Model Performance")
            st.code(classification_report(y_test, y_pred), language="text")

            st.subheader("🧪 Predict Dropout for New Students")
            uploaded_pred_file = st.file_uploader("📥 Upload Test File for Prediction (CSV)", type=["csv"], key="predict")

            if uploaded_pred_file:
                pred_df = pd.read_csv(uploaded_pred_file)
                pred_input = pred_df.copy()

                for col in pred_input.select_dtypes(include="object"):
                    pred_input[col] = le.fit_transform(pred_input[col].astype(str))

                pred_input = pred_input[X.columns]  # Ensure same column order
                predictions = model.predict(pred_input)
                pred_df["Dropout_Prediction"] = predictions

                st.write("🔍 Prediction Results")
                st.dataframe(pred_df)

                csv = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Prediction Results", data=csv, file_name="dropout_predictions.csv", mime="text/csv")

                # Save to SQLite
                conn = sqlite3.connect("database/dropout.db")
                pred_df.to_sql("predictions", conn, if_exists="replace", index=False)
                conn.close()
                st.success("✅ Predictions saved to SQLite database.")
    else:
        st.error("Training data not found. Please upload `students_train.csv`.")
