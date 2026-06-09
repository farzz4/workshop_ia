import streamlit as st
import numpy as np
import joblib

modele_clustering = joblib.load("modele_clustering.pkl")
modele_regression = joblib.load("modele_regression.pkl")
modele_classification = joblib.load("modele_classification.pkl")

st.title("Nature Harmony - Assistant de Sophie")


st.header("Profil du Visiteur")

age = st.slider("Âge du visiteur", 10, 80, 30)
taille_groupe = st.slider("Taille du groupe", 1, 10, 2)
nb_enfants = st.slider("Nombre d'enfants", 0, 5, 0)
score_physique = st.slider("Score physique (1-10)", 1, 10, 5)

if st.button("Prédire le profil visiteur"):
    visiteur = np.array([[age, taille_groupe, nb_enfants, score_physique]])
    groupe = modele_clustering.predict(visiteur)[0]
    
    noms_groupes = {
        0: "Famille",
        1: "Jeune Aventurier",
        2: "Adulte Tranquille"
    }
    st.success(f"Ce visiteur appartient au groupe : {noms_groupes[groupe]}")


    st.header("Conditions Météo du Jour")

est_weekend = st.selectbox("Est-ce un weekend ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
est_ferie = st.selectbox("Est-ce un jour férié ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
temperature = st.slider("Température (°C)", 0, 40, 20)
precipitation = st.slider("Précipitations (mm)", 0, 50, 0)
vacance_scolaire = st.selectbox("Vacances scolaires ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")


if st.button("Prédire le nombre de visiteurs et l'activité"):
    jour = np.array([[est_weekend, est_ferie, temperature, precipitation, vacance_scolaire]])
    
    nb_visiteurs = modele_regression.predict(jour)[0]
    activite = modele_classification.predict(jour)[0]
    
    st.success(f"Nombre de visiteurs attendus : {int(nb_visiteurs)}")
    st.success(f"Nombre d'employés nécessaires : {int(nb_visiteurs) // 20}")
    
    if activite == 1:
        st.success("Activité conseillée : Outdoor")
    else:
        st.success("Activité conseillée : Indoor")


#Si pas streamlit : pip install streamlit 
#Pour faire tourner l'appli : streamlit run app.py 