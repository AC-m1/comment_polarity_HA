from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline # preprocessing, model together
from sklearn.feature_extraction.text import TfidfVectorizer # to get numeric values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV # main addition of the model, the hyperparameter tuning


def _build_base_pipeline() -> Pipeline: #clean definition of classic model for GridSearchCV to take the pipeline and tweak parameters without repetitive rewrites
    """
    Base TF-IDF + Logistic Regression pipeline.
    Hyperparameters will be tuned via GridSearchCV.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_classic_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[Pipeline, float, float]:
    """
    Train a TF-IDF + Logistic Regression model with hyperparameter tuning.

    Returns
    -------
    best_model : Pipeline
        Pipeline fitted on the **full** dataset (train + test) using the best hyperparameters.
    acc : float
        Accuracy on the held-out test set (before refitting on full data).
    f1 : float
        Weighted F1 on the held-out test set (before refitting on full data).
    """
    print("\n=== TF-IDF + Logistic Regression (with hyperparameter tuning) ===") #print!

    # for safety: ensure string type. to deal with NaNs...
    X_train = train_df["text"].astype(str)
    y_train = train_df["label"]
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"]

    base_pipeline = _build_base_pipeline() # for starting from earlier defined pipeline

    # hyperparameter
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_df": [0.9, 1.0],
        "tfidf__min_df": [1, 3, 5],
        "tfidf__max_features": [5000, 10000, None],
        "tfidf__sublinear_tf": [True, False],
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"], # line 66 for a list of parameters against dominant parts of the log.
    }

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
        verbose=2, # to cross validate hyperparmeter search
    )

    print("\n[CLASSIC] Starting GridSearchCV over TF-IDF + LogReg hyperparameters...") # print!
    grid.fit(X_train, y_train)

    print("\n[CLASSIC] Best hyperparameters found:") # print to get best performaning parameter
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

    best_model = grid.best_estimator_

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    print("\n[CLASSIC] Evaluation on held-out test set")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"[CLASSIC] Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")

    # Also, to retrain on full dataset (train + test) with best hyperparameters
    print("\n[CLASSIC] Re-training best configuration on full dataset for saving...")
    full_X = pd.concat([X_train, X_test])
    full_y = pd.concat([y_train, y_test])
    best_model.fit(full_X, full_y)

    print("[CLASSIC] Done. Returning tuned model and test metrics.")
    return best_model, acc, f1
