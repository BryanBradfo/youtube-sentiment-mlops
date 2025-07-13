# tests/test_preprocess.py (CORRIGÉ)
import sys
import os

# Ajoute le dossier 'src' au chemin pour que Python puisse trouver 'preprocess'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from preprocess import clean_text


def test_clean_text():
    """Vérifie que le nettoyage de base fonctionne."""
    input_text = "Super vidéo @Squeezie ! Regardez ça https://t.co/xyz"
    # La nouvelle sortie attendue
    expected_output = "super vidéo regardez ça"
    assert clean_text(input_text) == expected_output


def test_strip_emojis_and_accents():
    """Vérifie que les emojis et les accents sont bien gérés."""
    input_text = "J'adore cette vidéo 😂"
    expected_output = "jadore cette vidéo <emoji>"
    assert clean_text(input_text) == expected_output
