import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_CSV = "data/processed/commentaires_sentiment.csv"

def load_data():
    df = pd.read_csv(DATA_CSV)
    X = df["commentaire_clean"]
    y = df["sentiment"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_and_train(X_train, y_train):
    tfidf = TfidfVectorizer(max_features=5_000, ngram_range=(1,2))
    X_vec = tfidf.fit_transform(X_train)

    # Modeles de base
    lr = LogisticRegression(max_iter=1_000)
    # On réutilise la colonne "sentiment" comme feature simple (proxy)
    # en mode stacking basique
    stack = StackingClassifier(
        estimators=[("lr", lr)],
        final_estimator=LogisticRegression(),
        passthrough=True
    )

    mlflow.set_experiment("sentiment-stack")
    with mlflow.start_run():
        mlflow.log_params({"vect_max_features": 5000})
        stack.fit(X_vec, y_train)
        mlflow.sklearn.log_model(stack, "stacking_model", registered_model_name="SentimentStack")
        return tfidf, stack

def evaluate(tfidf, model, X_test, y_test):
    X_vec = tfidf.transform(X_test)
    preds = model.predict(X_vec)
    print(classification_report(y_test, preds))
    mlflow.log_metric("accuracy", (preds == y_test).mean())

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    tfidf, model = build_and_train(X_train, y_train)
    evaluate(tfidf, model, X_test, y_test)
