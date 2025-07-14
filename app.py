import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import joblib  # Pour charger le modèle .pkl
import os
from src.fetch_comments import fetch_comments
from src.preprocess import clean_text
import matplotlib.pyplot as plt

# --- CONFIGURATION ET CHARGEMENT DU MODÈLE ---

REPO_URL = (
    "https://api.github.com/repos/BryanBradfo/youtube-sentiment-mlops/releases/latest"
)


# @st.cache_resource est crucial pour ne télécharger et charger le modèle qu'une seule fois.
@st.cache_resource
def download_and_load_model():
    """Télécharge, dézippe et charge le modèle depuis la dernière release GitHub."""
    st.info("Téléchargement du dernier modèle depuis GitHub Releases...")

    # Récupérer les informations de la dernière release
    response = requests.get(REPO_URL)
    response.raise_for_status()
    release_info = response.json()

    # Trouver l'URL de l'asset (release.zip)
    asset_url = release_info["assets"][0]["browser_download_url"]

    # Télécharger le fichier zip
    zip_response = requests.get(asset_url)
    zip_response.raise_for_status()

    # Dézipper en mémoire
    zip_file = zipfile.ZipFile(io.BytesIO(zip_response.content))

    # Extraire et charger le modèle
    # Le chemin dépend de la structure de votre zip. MLflow sauve le modèle dans un sous-dossier.
    model_path_in_zip = "mlruns/837460228489422544/models/m-5ff3f9180f9b4c5dbf7e371c06ef39b0/artifacts/model.pkl"

    with zip_file.open(model_path_in_zip) as model_file:
        model = joblib.load(model_file)

    st.success("Modèle chargé avec succès !")
    return model


# Charger le modèle
model = download_and_load_model()


# --- INTERFACE UTILISATEUR STREAMLIT ---

st.title("Analyseur de Sentiments des Commentaires YouTube")

video_url = st.text_input(
    "Collez l'URL de la vidéo YouTube ici :",
    "https://www.youtube.com/watch?v=ClF55GE7zPI",
)

if st.button("Analyser les commentaires"):
    if video_url:
        try:
            video_id = video_url.split("v=")[1].split("&")[0]
            api_key = st.secrets["YOUTUBE_API_KEY"]

            with st.spinner("Récupération et nettoyage des commentaires..."):
                # 1. Fetch and Preprocess
                df = fetch_comments(api_key, video_id)
                df["commentaire_clean"] = (
                    df["commentaire"].astype(str).apply(clean_text)
                )

            with st.spinner("Analyse des sentiments en cours... (inférence)"):
                # 2. Inference
                predictions = model.predict(df["commentaire_clean"])
                df["sentiment"] = predictions

            # 3. Display Results
            st.subheader("Résultats de l'Analyse")
            sentiment_counts = df["sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis(
                "equal"
            )  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

            st.write("Distribution des sentiments :")
            st.dataframe(sentiment_counts)

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
    else:
        st.warning("Veuillez entrer une URL de vidéo.")
