
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "parkinsons.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

FEATURES = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
TARGET = "status"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("parkinsons.csv not found. Place it next to this script.")

    df = pd.read_csv(CSV_PATH)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = SVC(kernel="linear", probability=True, random_state=2)
    model.fit(X_train_std, y_train)

    y_pred = model.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Saved:", MODEL_PATH, SCALER_PATH)

if __name__ == "__main__":
    main()
