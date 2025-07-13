import pandas as pd
from googleapiclient.discovery import build

# --- Configuration ---
# Remplacez par votre clé d'API obtenue sur Google Cloud Platform
API_KEY = "AIzaSyBG6QZPtHupQ07Q4_9-0J7__G48YhODNfI"
# Remplacez par l'identifiant de la vidéo YouTube
VIDEO_ID = "ClF55GE7zPI"
# Nom du fichier de sortie
OUTPUT_FILE = "commentaires_youtube.csv"

def recuperer_commentaires_youtube(api_key, video_id):
    """
    Récupère tous les commentaires d'une vidéo YouTube et les retourne sous forme de liste.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    commentaires = []

    try:
        # Première requête pour obtenir les premiers commentaires
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100  # Nombre de commentaires par page (max 100)
        )

        while request:
            response = request.execute()

            for item in response["items"]:
                commentaire = item["snippet"]["topLevelComment"]["snippet"]
                commentaires.append({
                    'auteur': commentaire['authorDisplayName'],
                    'commentaire': commentaire['textDisplay'],
                    'likes': commentaire['likeCount'],
                    'date_publication': commentaire['publishedAt']
                })

            # Vérifie s'il y a une page suivante de commentaires
            request = youtube.commentThreads().list_next(request, response)

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

    return commentaires

def sauvegarder_en_csv(data, filename):
    """
    Sauvegarde les données dans un fichier CSV.
    """
    if data:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Les commentaires ont été sauvegardés dans le fichier '{filename}'")
    else:
        print("Aucun commentaire n'a été récupéré.")

if __name__ == "__main__":
    print("Début de la récupération des commentaires...")
    liste_commentaires = recuperer_commentaires_youtube(API_KEY, VIDEO_ID)

    if liste_commentaires:
        print(f"{len(liste_commentaires)} commentaires récupérés.")
        sauvegarder_en_csv(liste_commentaires, OUTPUT_FILE)
    else:
        print("La récupération des commentaires a échoué.")