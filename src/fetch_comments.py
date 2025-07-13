import os
import pandas as pd
from googleapiclient.discovery import build

API_KEY    = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID   = "ClF55GE7zPI"
OUTPUT_CSV = "data/raw/commentaires_youtube.csv"

def fetch_comments(api_key: str, video_id: str) -> pd.DataFrame:
    yt = build("youtube", "v3", developerKey=api_key)
    all_comments = []
    req = yt.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )
    while req:
        res = req.execute()
        for item in res["items"]:
            s = item["snippet"]["topLevelComment"]["snippet"]
            all_comments.append({
                "auteur":          s["authorDisplayName"],
                "commentaire":     s["textDisplay"],
                "likes":           s["likeCount"],
                "date_publication":s["publishedAt"]
            })
        req = yt.commentThreads().list_next(req, res)
    return pd.DataFrame(all_comments)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = fetch_comments(API_KEY, VIDEO_ID)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"{len(df)} commentaires sauvegardés dans {OUTPUT_CSV}")
