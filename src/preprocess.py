import pandas as pd
import re
import emoji

INPUT_CSV  = "data/raw/commentaires_youtube.csv"
OUTPUT_CSV = "data/processed/commentaires_clean.csv"

def strip_emojis(text: str) -> str:
    # garde l'emoji sous forme token : ex. "😢" → "<EMOJI_1>"
    def replace(match):
        replace.counter += 1
        return f"<EMOJI_{replace.counter}>"
    replace.counter = 0
    return emoji.replace_emoji(text, replace)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+","", text)            # liens
    text = re.sub(r"@\w+","", text)               # mentions
    text = strip_emojis(text)
    text = re.sub(r"[^a-z0-9_<>\s]+","", text)    # caractères spéciaux
    return text.strip()

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df["commentaire_clean"] = df["commentaire"].astype(str).apply(clean_text)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Prétraitement OK → {OUTPUT_CSV}")
