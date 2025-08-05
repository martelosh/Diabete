import streamlit as st
import pandas as pd
import os

from utils.model_utils import load_best_model, predict_with_model

st.title("📝 Form di autovalutazione del rischio diabete")

st.markdown("Compila tutti i campi per simulare il tuo rischio.")

# Input utente (tranne BMI)
gender = st.selectbox("Sesso", [0, 1])
age = st.slider("Età", 18, 90)
highbp = st.selectbox("Hai la pressione alta?", [0, 1])
highchol = st.selectbox("Hai il colesterolo alto?", [0, 1])
cholcheck = st.selectbox("Hai controllato il colesterolo negli ultimi 5 anni?", [0, 1])
smoker = st.selectbox("Fumi?", [0, 1])
stroke = st.selectbox("Hai avuto un ictus?", [0, 1])
heartdisease = st.selectbox("Hai malattie cardiache?", [0, 1])
physactivity = st.selectbox("Fai attività fisica?", [0, 1])
fruits = st.selectbox("Mangi frutta regolarmente?", [0, 1])
veggies = st.selectbox("Mangi verdura regolarmente?", [0, 1])
hvyalcoh = st.selectbox("Bevi molto alcol?", [0, 1])
anyhealthcare = st.selectbox("Hai accesso a servizi sanitari?", [0, 1])
nomedicalcare = st.selectbox("Hai evitato cure per costi?", [0, 1])
genhlth = st.slider("Salute generale (1=Ottima, 5=Pessima)", 1, 5)
menthlth = st.slider("Giorni con problemi mentali (ultimi 30)", 0, 30)
physhlth = st.slider("Giorni con problemi fisici (ultimi 30)", 0, 30)
diffwalk = st.selectbox("Hai difficoltà a camminare?", [0, 1])
education = st.slider("Istruzione (1=elementari, 6=laurea)", 1, 6)
income = st.slider("Reddito (1=basso, 8=alto)", 1, 8)

peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0)
altezza_cm = st.number_input("Altezza (cm)", min_value=100.0, max_value=220.0)
altezza_m = altezza_cm / 100
bmi = peso / (altezza_m ** 2)
st.write(f"👉 Il tuo BMI calcolato è: **{bmi:.2f}**")

# Submit
if st.button("✅ Invia"):
    new_record = {
        "HighBP": highbp,
        "HighChol": highchol,
        "CholCheck": cholcheck,
        "BMI": round(bmi, 1),
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heartdisease,
        "PhysActivity": physactivity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": hvyalcoh,
        "AnyHealthcare": anyhealthcare,
        "NoDocbcCost": nomedicalcare,
        "GenHlth": genhlth,
        "MentHlth": menthlth,
        "PhysHlth": physhlth,
        "DiffWalk": diffwalk,
        "Sex": gender,
        "Age": age,
        "Education": education,
        "Income": income
    }

    new_df = pd.DataFrame([new_record])

    # 1. Predizione
    model, model_type = load_best_model()
    prediction = predict_with_model(model, model_type, new_df)

    st.markdown(f"🧪 Il modello predice: **{int(prediction[0])}** (0 = Nessun diabete, 1 = Pre-diabete, 2 = Diabete)")

    # 2. Feedback
    feedback = st.radio("Questo risultato è corretto?", ["Sì", "No"])
    if feedback == "Sì":
        label = int(prediction[0])
    else:
        label = st.selectbox("Inserisci il valore corretto:", [0, 1, 2])

    # 3. Salva record con etichetta
    new_df["Diabetes_012"] = label
    feedback_path = "data/training_feedback.csv"

    if os.path.exists(feedback_path):
        df_old = pd.read_csv(feedback_path)
        df_new = pd.concat([df_old, new_df], ignore_index=True)
    else:
        df_new = new_df

    df_new.to_csv(feedback_path, index=False)
    st.success("✅ I tuoi dati sono stati salvati per il retraining del modello.")
