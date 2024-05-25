import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx
from onnxruntime import InferenceSession


class Data_Processor:
    def __init__(self, data_path, test_split=0.2) -> None:
        self.data_path = data_path
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.test_split = test_split
        self.scaler = StandardScaler()

    def process_data(self) -> tuple:
        self.load_data()
        self.split_data()
        self.preprocess_data()
        return self.x_train, self.x_test, self.y_train, self.y_test

    def load_data(self) -> None:
        logging.info("Loading data...")
        data = pd.read_csv(self.data_path)
        self.X = data.drop("Class", axis=1)
        self.y = data["Class"]
        logging.info("Data loaded.")

    def split_data(self) -> None:
        logging.info("Splitting data...")
        (self.x_train, self.x_test, self.y_train, self.y_test) = train_test_split(
            self.X, self.y, test_size=self.test_split, random_state=42
        )
        logging.info("Data splitted.")

    def preprocess_data(self) -> None:
        logging.info("Preprocessing data...")
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        logging.info("Data preprocessed.")


class ModelTrainer:
    def __init__(self, x_train, y_train) -> None:
        self.model = None
        self.x_train = x_train
        self.y_train = y_train

    def train_model(self, n_estimators=50, random_state=42) -> RandomForestClassifier:
        logging.info("Training model...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.model.fit(self.x_train, self.y_train)
        logging.info("Model trained.")
        return self.model


class ModelEvaluator:
    def __init__(self, model: InferenceSession, x_test, y_test) -> None:
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self) -> None:
        logging.info("Evaluating model...")
        y_pred = self.model.run(None, {"X": self.x_test})[0]
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)

        logging.info(
            f"Model evaluation results: \n"
            f"Accuracy: {accuracy}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1}\n"
            f"ROC AUC Score: {roc_auc}\n"
        )

    def visualize_confusion_matrix(self) -> None:
        logging.info("Visualizing confusion matrix...")
        y_pred = self.model.run(None, {"X": self.x_test})[0]
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        logging.info("Confusion matrix visualized.")

class ModelManager:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.model = None
    
    def save_model(self, model, x_train) -> bool:
        logging.info("Saving model...")
        onnx_model = to_onnx(model, x_train)
        try:
            with open(self.model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except Exception as e:
            logging.exception(e)
            return False
        logging.info("Model saved.")
        return True
    
    def load_model(self) -> bool:
        logging.info("Loading model...")
        try:
            with open(self.model_path, "rb") as f:
                self.model = InferenceSession(f.read())
        except Exception as e:
            logging.exception(e)
            return False
        logging.info("Model loaded.")
        return True
    
    def get_prediction(self, data) -> int:
        logging.info("Predicting...")
        try:
            prediction = self.model.run(None, {"X": data})[0]
        except Exception as e:
            logging.exception(e)
            return None
        logging.info(f"Prediction: {prediction}")
        return prediction
    
    def get_model(self) -> InferenceSession:
        return self.model
    

def main():
    # Example usage
    data_processor = Data_Processor("./creditcard_2023.csv", test_split=0.2)
    x_train, x_test, y_train, y_test = data_processor.process_data()
    model_manager = ModelManager("model.onnx")
    
    if not model_manager.load_model():
        detector = ModelTrainer(x_train, y_train)
        model = detector.train_model()
        # Save the model in a onnx file
        model_manager.save_model(model, x_train)
    else:
        model_manager.get_prediction(x_test[0:1])
        model = model_manager.get_model()
    
    evaluator = ModelEvaluator(model, x_test, y_test)
    evaluator.evaluate_model()
    evaluator.visualize_confusion_matrix()
    
    pred = model_manager.get_prediction(x_test[0:1])[0]
    actual = y_test.iloc[0]
    print(f"Prediction: {pred} Actual: {actual}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        main()
    except Exception as e:
        logging.exception(e)
