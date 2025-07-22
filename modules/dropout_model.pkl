# File: modules/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

def train_and_save_model():
    df = pd.read_csv("data/students_train.csv")

    X = df.drop(columns=["student_id", "name", "dropout"])
    y = df["dropout"]

    # Encode categoricals
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/dropout_model.pkl")
    print("Model trained and saved!")

# Run this manually once
# train_and_save_model()
