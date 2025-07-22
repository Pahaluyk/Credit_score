import logging
import pandas as pd
import os
from zenml import step
from catboost import CatBoostClassifier
from model.model_dev import CatBoostModel
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> CatBoostClassifier:
    try:
        cat_features = x_train.select_dtypes(include=['object']).columns.tolist()

        model_wrapper = CatBoostModel()
        trained_model = model_wrapper.train(
            x_train,
            y_train,
            x_test,
            y_test,
            cat_features=cat_features,
            iterations=8000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=1,
            random_seed=42,
            eval_metric="Accuracy",
            loss_function="MultiClass",
            verbose=100,
            train_dir="catboost_logs"
        )

        # Логируем параметры модели
        mlflow.log_params(trained_model.get_params())

        # Сохраняем модель
        save_dir = "saved_model"
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, "credit_score_model.cbm")
        trained_model.save_model(model_file)

        logging.info(f"Модель сохранена в {model_file}")

        # Логируем модель в MLflow
        mlflow.catboost.log_model(trained_model, artifact_path="models")

        return trained_model

    except Exception as e:
        logging.error(f"Ошибка при обучении модели: {e}")
        raise e
