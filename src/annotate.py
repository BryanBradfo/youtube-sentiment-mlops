# src/annotate.py
import os
import pandas as pd
from transformers import AutoTokenizer, pipeline

INPUT_CSV  = "data/processed/commentaires_clean.csv"
OUTPUT_CSV = "data/processed/commentaires_sentiment.csv"

def annotate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # Charge le tokenizer lent pour éviter les erreurs de conversion
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        use_fast=False
    )
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer=tokenizer,
        device=0  # GPU
    )

    labels, scores = [], []
    for txt in df["commentaire_clean"].tolist():
        res = sentiment_pipe(txt[:512])[0]
        labels.append(res["label"])
        scores.append(res["score"])

    df["sentiment"] = labels
    df["score"]     = scores
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df = annotate_sentiment(df)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Annotation sentiment OK → {OUTPUT_CSV}")
