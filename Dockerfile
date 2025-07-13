# Étape 1: Utiliser une image de base avec Conda
FROM continuumio/miniconda3

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et créer l'environnement
# On installe la version CPU de PyTorch car c'est le cas le plus courant pour un déploiement sans GPU
COPY environment.yml ./
RUN conda env create -f environment.yml && \
    conda run -n sentiment-mlops pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copier tout le reste du projet
COPY . .

# Définir le point d'entrée pour lancer la pipeline.
# --no-capture-output est utile pour voir les prints de vos scripts en temps réel.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sentiment-mlops", "dvc", "repro"]