# app/main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.db_utils import create_db_engine


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

# Caricamento dataset
df = pd.read_csv("data/diabete_data.csv")

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
