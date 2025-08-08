# app/main.py

from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# --- Config Streamlit ---
st.set_page_config(page_title="Informazioni sul Diabete", layout="wide")

st.title("🩺 Cos'è il Diabete?")
st.markdown("""
Il diabete è una condizione cronica che colpisce il modo in cui il corpo gestisce il glucosio nel sangue.
Esistono vari tipi, ma i principali sono:

- **Diabete di tipo 1**: autoimmune, spesso diagnosticato in giovane età.
- **Diabete di tipo 2**: legato a fattori come stile di vita e obesità.
- **Gestazionale**: può comparire durante la gravidanza.

👉 Fai attenzione ai sintomi come: sete eccessiva, minzione frequente, stanchezza, perdita di peso.
""")

st.subheader("📊 Esplora i dati")

# --- Percorsi ---
# app/main.py -> app/ (parent) -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = PROJECT_ROOT / "src" / "db" / "diabetes.db"
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"

# --- Data loading (cached) ---
@st.cache_data(ttl=300)
def load_data():
    engine = create_engine(SQLITE_URL)
    # Se la tabella si chiama diversamente, cambia qui:
    return pd.read_sql_table("diabetes_data", con=engine)

try:
    df = load_data()
except ValueError as e:
    st.error(f"Errore nel leggere la tabella: {e}")
    st.stop()
except Exception as e:
    st.error(f"Impossibile connettersi al database '{SQLITE_PATH}': {e}")
    st.stop()

# --- Grafico 1: Distribuzione BMI ---
st.write("### Distribuzione del BMI")
fig1, ax1 = plt.subplots()
df["BMI"].hist(bins=30, ax=ax1)
st.pyplot(fig1)

# --- Grafico 2: Distribuzione per età ---
st.write("### Distribuzione dell'età")
fig2, ax2 = plt.subplots()
df["Age"].hist(bins=20, ax=ax2)
st.pyplot(fig2)

# --- Navigazione al form (assicurati che esista la pagina) ---
# Per Streamlit multipage, la cartella deve essere app/pages/...
# e il file deve chiamarsi "form_di_autovalutazione.py"
if st.button("🧾 Fai il tuo test"):
    st.switch_page("pages/form_di_autovalutazione.py")
