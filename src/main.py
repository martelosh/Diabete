import os
from pathlib import Path
from dotenv import load_dotenv
from model_training import split_data, evaluate_models_cross_validation, tune_keras_model
from data_preprocessing import create_db_engine, test_connection, load_data_from_csv, preprocess_data, import_dataframe_to_db
from grid_search import run_grid_search_and_save, param_grids
import pickle
from keras.models import load_model
import accuracy_score

# === SETUP PROGETTO E PATH ===
project_root = Path(__file__).resolve().parent.parent  # esce da src e va alla root
env_path = project_root / '.env'

load_dotenv(dotenv_path=env_path)
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")
host = os.getenv("SQL_HOST")
port = 3306
database = os.getenv("SQL_DATABASE")
    
csv_path = project_root / "data" / "diabete_data.csv"
models_path = project_root / "data" / "grid_search_results"
os.makedirs(models_path, exist_ok=True)

# === CONNESSIONE AL DATABASE ===
engine = create_db_engine(username, password, host, port, database)
test_connection(engine)

# === IMPORTAZIONE DATI E PREPROCESSING ===
df = load_data_from_csv(csv_path)
df_clean = preprocess_data(df)

# === SPLIT TRAIN/TEST ===
x_train, x_test, y_train, y_test = split_data(df_clean, target_column="Diabetes_012")

# === CROSS-VALIDATION MODELLI CLASSICI ===
results, best_estimator, best_model_name = evaluate_models_cross_validation(x_train, y_train)
print("Risultati della cross-validazione:", results)

# === GRID SEARCH PER MODELLI CLASSICI ===
run_grid_search_and_save(
    estimator=best_estimator,
    param_grid=param_grids[best_model_name],
    x_train=x_train,
    y_train=y_train,
    model_name=best_model_name
)

# === TRAINING E SALVATAGGIO MODELLO KERAS ===
keras_model, val_acc_keras = tune_keras_model(x_train, y_train, x_test, y_test)
keras_model_path = models_path / "best_keras_model.h5"
keras_model.save(keras_model_path)
print(f"Modello Keras salvato in: {keras_model_path}")
print(f"Accuratezza di validazione Keras: {val_acc_keras:.4f}")

# === CONFRONTO TRA MODELLI SALVATI ===

# Carica modello scikit-learn
pkl_model_path = models_path / f"{best_model_name}_optimized_model.pkl"
with open(pkl_model_path, "rb") as f:
    sklearn_model = pickle.load(f)

# Carica modello Keras
keras_model_loaded = load_model(keras_model_path)

# Valutazione su test set
y_pred_sklearn = sklearn_model.predict(x_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

y_pred_keras = keras_model_loaded.predict(x_test)
y_pred_keras_classes = y_pred_keras.argmax(axis=1)
acc_keras = accuracy_score(y_test, y_pred_keras_classes)

# Stampa confronto
print(f"\n🔎 Accuratezza finale - Modello scikit-learn ({best_model_name}): {acc_sklearn:.4f}")
print(f"🔎 Accuratezza finale - Modello Keras: {acc_keras:.4f}")

if acc_sklearn > acc_keras:
    print(f"\n✅ Il miglior modello in assoluto è: {best_model_name} (scikit-learn) con accuratezza {acc_sklearn:.4f}")
else:
    print(f"\n✅ Il miglior modello in assoluto è: Rete neurale Keras con accuratezza {acc_keras:.4f}")