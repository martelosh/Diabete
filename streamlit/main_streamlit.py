# app/main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.db_utils import create_db_engine
import sys
import os


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

from sqlalchemy import create_engine

# 🔹 Rende visibile la cartella principale "Diabete" per importare da src/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from src.db_utils import create_db_engine  # se vuoi ancora usare funzioni MySQL

# 🔹 Connessione a SQLite (modifica il path se serve)
sqlite_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'db', 'diabetes.db')
sqlite_url = f"sqlite:///{sqlite_path}"

# Crea l'engine
engine = create_engine(sqlite_url)

# Legge la tabella in DataFrame
df = pd.read_sql_table("diabetes_data", con=engine)

# Grafico 1: Distribuzione BMI
st.write("### Distribuzione del BMI")
fig1, ax1 = plt.subplots()
df["BMI"].hist(bins=30, ax=ax1)
st.pyplot(fig1)

# Grafico 2: Distribuzione per età
st.write("### Distribuzione dell'età")
fig2, ax2 = plt.subplots()
df["Age"].hist(bins=20, ax=ax2)
st.pyplot(fig2)

# Pulsante per passare al form
if st.button("🧾 Fai il tuo test"):
    st.switch_page("pages/form_di_autovalutazione.py")
