import streamlit as st
import requests
import zipfile
import io
import joblib  # Pour charger le modèle .pkl
from src.fetch_comments import fetch_comments
from src.preprocess import clean_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

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
    model_path_in_zip = "mlruns/637075048534979660/models/m-d4051bb8019842afb077dbb9315535c8/artifacts/model.pkl"
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

            st.subheader("Nuage de Mots-Clés")
            with st.spinner("Génération du nuage de mots..."):
                # Joindre tous les commentaires en un seul grand texte
                full_text = " ".join(comment for comment in df.commentaire_clean)

                # Définir les mots à ignorer (stopwords) en français
                stopwords = set(STOPWORDS)
                stopwords.update(
                    [
                        "le",
                        "la",
                        "les",
                        "de",
                        "des",
                        "du",
                        "et",
                        "est",
                        "il",
                        "elle",
                        "on",
                        "un",
                        "une",
                        "que",
                        "qui",
                        "pour",
                        "pas",
                        "plus",
                        "cest",
                        "jai",
                        "vraiment",
                        "ça",
                        "fait",
                        "être",
                        "avoir",
                        "emoji",
                    ]
                )
                wordcloud = WordCloud(
                    stopwords=stopwords, background_color="white", width=800, height=400
                ).generate(full_text)

                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wordcloud, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

            # --- NOUVELLE SECTION 2 : AFFICHAGE D'EXEMPLES DE COMMENTAIRES ---
            st.subheader("Exemples de Commentaires par Sentiment")

            df["sentiment_lower"] = df["sentiment"].str.lower()

            # Créer des onglets pour chaque sentiment
            tab_pos, tab_neu, tab_neg = st.tabs(
                ["👍 Positifs", "😐 Neutres", "👎 Négatifs"]
            )

            with tab_pos:
                st.write("Top 10 des commentaires jugés positifs :")
                df_pos = df[df["sentiment_lower"] == "positive"]
                if not df_pos.empty:
                    # AMÉLIORÉ : Tri par likes et affichage du top 10
                    top_10_pos = df_pos.sort_values(by="likes", ascending=False).head(
                        10
                    )
                    st.dataframe(
                        top_10_pos[["auteur", "commentaire", "likes"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("Aucun commentaire positif trouvé.")

            with tab_neu:
                st.write("Top 10 des commentaires jugés neutres :")
                df_neu = df[df["sentiment_lower"] == "neutral"]
                if not df_neu.empty:
                    top_10_neu = df_neu.sort_values(by="likes", ascending=False).head(
                        10
                    )
                    st.dataframe(
                        top_10_neu[["auteur", "commentaire", "likes"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("Aucun commentaire neutre trouvé.")

            with tab_neg:
                st.write("Top 10 des commentaires jugés négatifs :")
                df_neg = df[df["sentiment_lower"] == "negative"]
                if not df_neg.empty:
                    top_10_neg = df_neg.sort_values(by="likes", ascending=False).head(
                        10
                    )
                    st.dataframe(
                        top_10_neg[["auteur", "commentaire", "likes"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("Aucun commentaire négatif trouvé.")

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
    else:
        st.warning("Veuillez entrer une URL de vidéo.")
