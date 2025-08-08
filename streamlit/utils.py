import os
import pickle
from pathlib import Path
import pandas as pd

# ✅ Evita crash all'import del modulo
try:
    from tensorflow.keras.models import load_model  # lazy use below
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

models_path = Path(__file__).resolve().parent.parent / "data" / "grid_search_results"
keras_model_path = models_path / "best_keras_model.h5"
pkl_model_path = next(models_path.glob("*_optimized_model.pkl"), None)  # qualunque best sklearn

def load_best_model():
    """
    Prova prima sklearn; opzionale: Keras solo se disponibile e file presente.
    """
    if pkl_model_path and pkl_model_path.exists():
        with open(pkl_model_path, "rb") as f:
            return ("sklearn", pickle.load(f))

    if TF_AVAILABLE and keras_model_path.exists():
        # import on demand, così non esplodiamo quando TF non c'è
        from tensorflow.keras.models import load_model as _load_model
        return ("keras", _load_model(keras_model_path))

    raise FileNotFoundError("Nessun modello trovato in data/grid_search_results")

def predict_with_model(model_tuple, X_df):
    kind, model = model_tuple
    if kind == "sklearn":
        return model.predict(X_df)
    elif kind == "keras":
        import numpy as np
        preds = model.predict(X_df)
        return np.argmax(preds, axis=1)
    else:
        raise ValueError("Tipo modello non riconosciuto")
