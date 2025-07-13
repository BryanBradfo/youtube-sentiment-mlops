import os
import pandas as pd
import re
import emoji

INPUT_CSV  = "data/raw/commentaires_youtube.csv"
OUTPUT_CSV = "data/processed/commentaires_clean.csv"

def strip_emojis(text: str) -> str:
    counter = {"i": 0}
    def replace_func(emj_char, emj_data):
        counter["i"] += 1
        return f"<EMOJI_{counter['i']}>"
    return emoji.replace_emoji(text, replace_func)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = strip_emojis(text)
    text = re.sub(r"[^a-z0-9_<>\s]+", "", text)
    return text.strip()

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df["commentaire_clean"] = df["commentaire"].astype(str).apply(clean_text)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Prétraitement OK → {OUTPUT_CSV}")
