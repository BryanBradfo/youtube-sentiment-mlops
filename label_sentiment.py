import pandas as pd
import torch
from transformers import pipeline
import emoji
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "commentaires_youtube.csv"
OUTPUT_FILE = "commentaires_youtube_sentiments.csv"
# Le traitement par lots est crucial pour ne pas saturer la VRAM de 8GB.
# Ajustez cette valeur si vous rencontrez des erreurs de mémoire.
BATCH_SIZE = 32 

# --- Fonctions ---

def preprocess_comment(text):
    """
    Nettoie et prépare le texte du commentaire.
    Convertit les emojis en leur description textuelle.
    """
    if not isinstance(text, str):
        return ""
    # Convertit les emojis en texte (ex: "😂" -> ":face_with_tears_of_joy:")
    return emoji.demojize(text, language='fr')

def map_sentiment_labels(label):
    """
    Traduit les labels du modèle en catégories simples.
    Le modèle 'nlptown/bert-base-multilingual-uncased-sentiment' retourne des étoiles.
    """
    if label in ["1 star", "2 stars"]:
        return "Négatif"
    elif label == "3 stars":
        return "Neutre"
    elif label in ["4 stars", "5 stars"]:
        return "Positif"
    return "Indéterminé"


def main():
    """
    Fonction principale pour charger les données, analyser les sentiments et sauvegarder.
    """
    # 1. Vérifier la disponibilité du GPU
    if not torch.cuda.is_available():
        print("Attention : Le GPU n'est pas disponible. Le traitement se fera sur CPU et sera très lent.")
        device = -1 # Utiliser le CPU
    else:
        print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
        device = 0 # Utiliser le premier GPU

    # 2. Charger les données
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Fichier '{INPUT_FILE}' chargé. {len(df)} commentaires à analyser.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{INPUT_FILE}' n'a pas été trouvé.")
        return

    # 3. Préparer le modèle
    print("Chargement du modèle d'analyse de sentiments...")
    # Modèle multilingue, léger et efficace, parfait pour notre cas d'usage.
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device
    )
    print("Modèle chargé.")

    # 4. Pré-traiter les commentaires
    print("Pré-traitement des commentaires (conversion des emojis)...")
    commentaires = df['commentaire'].apply(preprocess_comment).tolist()

    # 5. Analyser les sentiments par lots (batching)
    print(f"Début de l'analyse des sentiments par lots de {BATCH_SIZE}...")
    all_results = []
    # tqdm ajoute une barre de progression très pratique
    for i in tqdm(range(0, len(commentaires), BATCH_SIZE), desc="Analyse des lots"):
        batch = commentaires[i:i + BATCH_SIZE]
        results = sentiment_pipeline(batch)
        all_results.extend(results)

    # 6. Ajouter les résultats au DataFrame
    sentiments = [res['label'] for res in all_results]
    scores = [res['score'] for res in all_results]
    
    df['sentiment_label'] = sentiments
    df['sentiment_score'] = scores
    df['sentiment_categorie'] = df['sentiment_label'].apply(map_sentiment_labels)

    # 7. Sauvegarder le nouveau fichier CSV
    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nAnalyse terminée ! Les résultats ont été sauvegardés dans '{OUTPUT_FILE}'.")
        print("\nAperçu des résultats :")
        print(df[['commentaire', 'sentiment_categorie', 'sentiment_score']].head())
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")


if __name__ == "__main__":
    main()