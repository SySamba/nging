import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

# Appliquer un style CSS personnalisé
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2rem;
        color: #4A90E2;
    }
    .success {
        font-weight: bold;
        color: green;
        font-size: 1.2rem;
    }
    .stTextInput {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction de modèle bi-exponentiel
def biexponential(t, A, alpha, B, beta):
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

# Fonction pour ajuster le modèle et calculer la clairance
def calculate_cl(time, dv, dose):
    try:
        A_initial, B_initial = dv[0], dv[-1]
        alpha_initial = np.log(dv[0] / dv[1]) / (time[1] - time[0])
        beta_initial = np.log(dv[-2] / dv[-1]) / (time[-1] - time[-2])
        initial_guess = (A_initial, alpha_initial, B_initial, beta_initial)

        popt, _ = curve_fit(biexponential, time, dv, p0=initial_guess, maxfev=90000)
        A, alpha, B, beta = popt

        # Calcul de l'AUC et de la clairance
        auc = np.abs((A / alpha) + (B / beta))
        cl = dose / auc

        return cl
    except Exception as e:
        st.error(f"Erreur de calcul : {str(e)}")
        return None

# Interface utilisateur Streamlit
def main():
    st.markdown('<h1 class="title">💊 Analyse de Clairance Pharmacocinétique</h1>', unsafe_allow_html=True)

    st.write("""
    Cette application permet de calculer la clairance d'un médicament à partir des données de concentration.
    **Entrez vos valeurs ci-dessous :**
    """)

    # Inputs avec une disposition en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        dose = st.number_input("💉 Dose du médicament (mg)", min_value=0.0, value=100.0, step=1.0)
    
    with col2:
        patient_id = st.text_input("🆔 ID du patient", "Patient_001")

    uploaded_data = st.text_area("📋 Collez vos données ici (Format: `Temps, Concentration`)")

    # Traitement des données
    if uploaded_data:
        try:
            data = [list(map(float, line.split(','))) for line in uploaded_data.strip().split('\n')]
            time, dv = np.array([x[0] for x in data]), np.array([x[1] for x in data])

            if len(time) > 1:
                cl = calculate_cl(time, dv, dose)
                if cl is not None:
                    st.markdown(f'<p class="success">✅ La clairance calculée pour {patient_id} est : <strong>{cl:.2f} L/h</strong></p>', unsafe_allow_html=True)

                    # Amélioration du graphique
                    fig, ax = plt.subplots(figsize=(7, 4))
                    sns.set_style("whitegrid")
                    sns.lineplot(x=time, y=dv, marker="o", color="#E74C3C", ax=ax)

                    ax.set_xlabel("Temps (h)", fontsize=12)
                    ax.set_ylabel("Concentration (mg/L)", fontsize=12)
                    ax.set_title("📊 Concentration en fonction du temps", fontsize=14)
                    ax.legend(["Concentration"], loc="best")
                    st.pyplot(fig)
            else:
                st.error("Veuillez entrer au moins 2 points de données.")
        except ValueError:
            st.error("Erreur : assurez-vous que vos données sont bien formatées (ex: `1, 10.5`).")

# Exécution de l'application
if __name__ == "__main__":
    main()
