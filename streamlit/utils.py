# utils/model_utils.py

import pickle
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model

models_path = Path(__file__).resolve().parent.parent / "data" / "grid_search_results"
keras_model_path = models_path / "best_keras_model.h5"

def load_best_model():
    """
    Carica il miglior modello salvato (.pkl o .h5), preferendo quello specificato in best_model.txt
    """
    try:
        with open(models_path / "best_model.txt", "r") as f:
            model_type = f.read().strip()
    except FileNotFoundError:
        model_type = "keras"

    if model_type == "sklearn":
        pkl_file = next(models_path.glob("*_optimized_model.pkl"), None)
        if pkl_file:
            model = pickle.load(open(pkl_file, "rb"))
            return model, "sklearn"
    else:
        model = load_model(keras_model_path)
        return model, "keras"

def predict_with_model(model, model_type, df: pd.DataFrame):
    if model_type == "sklearn":
        return model.predict(df)
    else:
        probs = model.predict(df)
        return probs.argmax(axis=1)
