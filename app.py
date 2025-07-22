import streamlit as st
import pandas as pd
import joblib
from modules.database import init_db, save_prediction, get_all_predictions
from modules.interventions import recommend_intervention

# Initialize DB
init_db()

# Load model
model = joblib.load("models/dropout_model.pkl")

st.set_page_config(page_title="Dropout Prediction System", layout="wide")
st.title("🎓 AI-Powered Student Dropout Prediction & Retention System")

menu = ["Upload & Predict", "Admin Dashboard"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Upload & Predict":
    st.subheader("📤 Upload Student Data for Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        X = df.drop(columns=["student_id", "name"])
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype("category").cat.codes

        predictions = model.predict_proba(X)[:, 1]
        df["Dropout_Risk"] = predictions
        df["Recommendation"] = df["Dropout_Risk"].apply(recommend_intervention)

        for i in df.index:
            save_prediction(df.loc[i, "student_id"],
                            df.loc[i, "name"],
                            df.loc[i, "Dropout_Risk"],
                            df.loc[i, "Recommendation"])

        st.dataframe(df[["student_id", "name", "Dropout_Risk", "Recommendation"]])
        st.success("✅ Predictions saved to database!")

elif choice == "Admin Dashboard":
    st.subheader("📊 Dropout Monitoring Dashboard")

    data = get_all_predictions()
    if data:
        df = pd.DataFrame(data, columns=["Student ID", "Name", "Risk", "Intervention"])
        st.dataframe(df)

        st.bar_chart(df["Risk"])
        high_risk = df[df["Risk"] > 0.75]
        st.warning(f"⚠️ {len(high_risk)} high-risk students detected!")

        st.download_button("Download Full Report", df.to_csv(index=False), "report.csv", "text/csv")
    else:
        st.info("No predictions yet.")
