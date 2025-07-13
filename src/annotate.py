# src/annotate.py
import os
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, pipeline

INPUT_CSV  = "data/processed/commentaires_clean.csv"
OUTPUT_CSV = "data/processed/commentaires_sentiment.csv"

def annotate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # Charge MANUELLEMENT le tokenizer et le modèle lents
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0
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
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df = annotate_sentiment(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Annotation sentiment OK → {OUTPUT_CSV}")
