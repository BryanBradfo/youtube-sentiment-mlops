import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import os

# Chemin du dataset annoté
DATA_CSV = "data/processed/commentaires_sentiment.csv"

# Fichier pour le rapport de performance
METRICS_FILE = "reports/metrics.txt"


def load_data():
    # Chargement et nettoyage des NaN
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["commentaire_clean", "sentiment"])
    X = df["commentaire_clean"]
    y = df["sentiment"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_and_train(X_train, y_train):
    # TF-IDF vectorisation
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = tfidf.fit_transform(X_train)

    # Modèle de base et stacking
    lr = LogisticRegression(max_iter=1000)
    stack = StackingClassifier(
        estimators=[("lr", lr)], final_estimator=LogisticRegression(), passthrough=True
    )

    model_pipeline = Pipeline([("tfidf", tfidf), ("classifier", stack)])

    mlflow.set_experiment("sentiment-stack")
    with mlflow.start_run():
        # Log des paramètres
        mlflow.log_params(
            {
                "vect_max_features": 5000,
                "ngram_range": "(1,2)",
                "base_model": "LogisticRegression",
            }
        )
        # Entraînement
        model_pipeline.fit(X_vec, y_train)
        # Enregistrement du modèle dans MLflow et registre
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="sentiment_pipeline",
            registered_model_name="SentimentStack",
        )
        return model_pipeline


def evaluate(model, X_test, y_test):
    # Évaluation
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    print(report)

    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        f.write(report)

    mlflow.log_metric("accuracy", float((preds == y_test).mean()))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    # L'entraînement retourne la pipeline complète
    model = build_and_train(X_train, y_train)
    # L'évaluation utilise directement la pipeline
    evaluate(model, X_test, y_test)
    print("Entraînement et évaluation terminés.")
