import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ClassificationEvaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        pass


class Accuracy(ClassificationEvaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        try:
            logging.info("Entered the calculate_score method of the Accuracy class")
            acc = accuracy_score(y_true, y_pred)
            logging.info("Accuracy score: " + str(acc))
            return acc
        except Exception as e:
            logging.error("Exception in Accuracy class: " + str(e))
            raise e


class Precision(ClassificationEvaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        try:
            logging.info("Entered the calculate_score method of the Precision class")
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            logging.info("Precision score: " + str(precision))
            return precision
        except Exception as e:
            logging.error("Exception in Precision class: " + str(e))
            raise e


class Recall(ClassificationEvaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        try:
            logging.info("Entered the calculate_score method of the Recall class")
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            logging.info("Recall score: " + str(recall))
            return recall
        except Exception as e:
            logging.error("Exception in Recall class: " + str(e))
            raise e


class F1Score(ClassificationEvaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        try:
            logging.info("Entered the calculate_score method of the F1Score class")
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            logging.info("F1 score: " + str(f1))
            return f1
        except Exception as e:
            logging.error("Exception in F1Score class: " + str(e))
            raise e


class ROCAUC(ClassificationEvaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        try:
            logging.info("Entered the calculate_score method of the ROCAUC class")
            if y_proba is None:
                raise ValueError("y_proba must be provided for ROC AUC calculation")
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            logging.info("ROC AUC score: " + str(auc))
            return auc
        except Exception as e:
            logging.error("Exception in ROCAUC class: " + str(e))
            return 0.0
