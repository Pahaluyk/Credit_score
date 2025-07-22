import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test, cat_features):
        pass


class CatBoostModel(Model):
    def train(self, x_train, y_train, x_test, y_test, **kwargs):
        model = CatBoostClassifier(
            **kwargs
        )
        model.fit(x_train, y_train, eval_set=(x_test, y_test))
        return model

    def optimize(self, trial, x_train, y_train, x_test, y_test, cat_features):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": 42,
        }

        model = self.train(x_train, y_train, cat_features=cat_features, **params)
        y_pred_proba = model.predict_proba(x_test)

        try:
            score = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        except Exception as e:
            logging.warning(f"ROC AUC computation failed: {e}")
            score = 0.0

        return score
