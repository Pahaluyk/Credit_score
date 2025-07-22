import logging
import numpy as np
import pandas as pd
from model.evaluation import Accuracy, Precision, Recall, F1Score, ROCAUC
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from catboost import CatBoostClassifier
import mlflow
from zenml.client import Client

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tempfile
import os

experiment_tracker = Client().active_stack.experiment_tracker

def log_feature_importance_plot(importances_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importances_df.sort_values(by="Importances", ascending=False),
        x="Importances",
        y="Feature Id",
        palette="viridis"
    )
    plt.title("Feature Importance (CatBoost)")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp_dir:
        fi_path = os.path.join(tmp_dir, "feature_importance.png")
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: CatBoostClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
    Annotated[float, "f1_score"],
    Annotated[float, "roc_auc"]
]:
    try:
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)

        acc = Accuracy().calculate_score(y_test, y_pred)
        precision = Precision().calculate_score(y_test, y_pred)
        recall = Recall().calculate_score(y_test, y_pred)
        f1 = F1Score().calculate_score(y_test, y_pred)
        roc_auc = ROCAUC().calculate_score(y_test, y_pred, y_proba)

        mlflow.log_metrics({
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        })

        # Логгируем параметры модели
        mlflow.log_params(model.get_params())

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path, artifact_path="plots")

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
            report_df.to_csv(f.name)
            mlflow.log_artifact(f.name, artifact_path="reports")

        # Feature Importance
        importances_df = model.get_feature_importance(prettified=True)
        print(importances_df.columns)
        print(importances_df.head())

        log_feature_importance_plot(importances_df)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
            importances_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, artifact_path="tables")

        return acc, precision, recall, f1, roc_auc

    except Exception as e:
        logging.error(e)
        raise e
