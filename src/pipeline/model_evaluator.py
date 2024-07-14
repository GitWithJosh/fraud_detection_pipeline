import logging

import numpy as np
from onnxruntime import InferenceSession
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

class ModelEvaluator:
    """
    Class for evaluating and visualizing the performance of a machine learning model.

    Args:
        model (InferenceSession): The trained machine learning model.
        x_test: The input features for testing the model.
        y_test: The true labels for testing the model.

    Attributes:
        model (InferenceSession): The trained machine learning model.
        x_test: The input features for testing the model.
        y_test: The true labels for testing the model.
    """

    def __init__(self, model: InferenceSession, x_test, y_test) -> None:
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self) -> None:
        """
        Evaluates the performance of the model by calculating various metrics.

        Prints the accuracy, precision, recall, F1 score, and ROC AUC score of the model.
        """
        logging.info("Evaluating model...")
        # Ensure the input data is of type float32
        self.x_test = self.x_test.astype(np.float32)
        output = self.model.run(None, {"X": self.x_test})[0]
        y_pred = np.where(output > 0.5, 1, 0).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)

        logging.info(
            f"Model evaluation results:\n"
            f"Accuracy: {accuracy}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1}\n"
            f"ROC AUC Score: {roc_auc}"
        )
    
    def get_evaluation_metrics(self) -> dict:
        """
        Evaluates the performance of the model by calculating various metrics.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        logging.info("Evaluating model...")
        # Ensure the input data is of type float32
        self.x_test = self.x_test.astype(np.float32)
        output = self.model.run(None, {"X": self.x_test})[0]
        y_pred = np.where(output > 0.5, 1, 0).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)

        logging.info("Model evaluated.")
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC Score": roc_auc,
        }

    def visualize_confusion_matrix(self) -> np.ndarray:
        """
        Visualizes the confusion matrix of the model's predictions.

        Returns:
            np.ndarray: The confusion matrix.
        """
        logging.info("Visualizing confusion matrix...")
        # Ensure the input data is of type float32
        self.x_test = self.x_test.astype(np.float32)
        output = self.model.run(None, {"X": self.x_test})[0]
        y_pred = np.where(output > 0.5, 1, 0).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        logging.info("Confusion matrix visualized.")
        return cm