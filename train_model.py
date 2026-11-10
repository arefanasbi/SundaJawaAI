from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "data_bahasa.csv"
MODEL_PATH = ROOT / "models" / "model_bahasa.pkl"

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    if df["label"].nunique() < 2:
        raise ValueError("Dataset harus memiliki minimal dua label bahasa.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Akurasi data uji: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (urutan label):", model.classes_)
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel tersimpan: {MODEL_PATH}")

if __name__ == "__main__":
    main()
