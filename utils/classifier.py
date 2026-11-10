from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_bahasa.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model belum dilatih. Jalankan python train_model.py.")
    return joblib.load(MODEL_PATH)

def predict_language(model, text):
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])
    # Threshold sederhana untuk menghindari prediksi terlalu yakin pada input yang meragukan.
    label = str(classes[best_index]) if confidence >= 0.48 else "Tidak yakin"
    scores = {str(classes[i]): float(probabilities[i]) for i in range(len(classes))}
    return label, confidence, scores
