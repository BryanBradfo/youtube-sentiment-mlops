import pandas as pd
from transformers import pipeline

INPUT_CSV  = "data/processed/commentaires_clean.csv"
OUTPUT_CSV = "data/processed/commentaires_sentiment.csv"

def annotate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # modèle multi-langue adapté aux emojis
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=0  # 0 pour GPU
    )
    labels, scores = [], []
    for txt in df["commentaire_clean"].tolist():
        res = sentiment(txt[:512])[0]
        labels.append(res["label"])
        scores.append(res["score"])
    df["sentiment"] = labels
    df["score"]     = scores
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df = annotate_sentiment(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Annotation sentiment OK → {OUTPUT_CSV}")
