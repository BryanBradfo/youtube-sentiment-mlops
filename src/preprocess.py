# src/preprocess.py (CORRIGÉ)

import os
import pandas as pd
import re
import emoji

INPUT_CSV = "data/raw/commentaires_youtube.csv"
OUTPUT_CSV = "data/processed/commentaires_clean.csv"


def strip_emojis(text: str) -> str:
    # On met directement le préfixe en minuscules pour être cohérent
    def replace_func(emj_char, emj_data):
        return " <emoji> "  # On ajoute des espaces pour séparer l'emoji des mots

    return emoji.replace_emoji(text, replace_func)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Retire les URL
    text = re.sub(r"@\w+", "", text)  # Retire les mentions

    # On remplace les emojis AVANT de supprimer la ponctuation
    text = strip_emojis(text)

    # Autorise les caractères alphanumériques (y compris les accents) et l'espace.
    # On garde aussi '<' et '>' pour nos balises emoji.
    # Le 'à-ÿ' couvre la plupart des caractères accentués français.
    text = re.sub(r"[^a-z0-9à-ÿ<>_\s]", "", text)

    # Retire les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df["commentaire_clean"] = df["commentaire"].astype(str).apply(clean_text)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Prétraitement OK → {OUTPUT_CSV}")
