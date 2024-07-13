from enum import Enum
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from onnxruntime import InferenceSession
from onnx import ModelProto

from tensorflow import TensorSpec, float32
from skl2onnx.common.data_types import FloatTensorType

class ModelType(Enum):
    RandomForest = 1
    NeuralNetwork = 2
    XGBoost = 3
    GBC = 4

class Model:
    """
    Represents a machine learning model abstracted from different model types.

    This class encapsulates the model initialization, training, and conversion to ONNX format,
    supporting multiple types of models including RandomForest, NeuralNetwork, XGBoost, and GBC.

    Attributes:
        modeltype (ModelType): The type of the model.
        model (object): The instantiated model object.

    Methods:
        initialize_model: Initializes the model based on its type.
        train_model: Trains the model using the provided training data.
        convert_to_onnx: Converts the model to ONNX format for interoperability.
        get_model: Returns the instantiated model object.
    """
    def __init__(self, modeltype: ModelType):
        self.modeltype = modeltype
        self.model = self.initialize_model()

    def initialize_model(self) -> object:
        """
        Initializes the model based on its type.

        Returns:
            object: The instantiated model object.

        Raises:
            ValueError: If the model type is unsupported.
        """
        if self.modeltype == ModelType.RandomForest:
            return self._init_random_forest()
        elif self.modeltype == ModelType.NeuralNetwork:
            return self._init_neural_network()
        elif self.modeltype == ModelType.XGBoost:
            return self._init_xgboost()
        elif self.modeltype == ModelType.GBC:
            return self._init_gbc()
        else:
            raise ValueError("Unsupported model type")

    def train_model(self, x_train, y_train) -> None:
        """
        Trains the model using the provided training data.

        Parameters:
            x_train: The training data features.
            y_train: The training data labels.
        """
        if self.modeltype == ModelType.NeuralNetwork:
            self.model.fit(x_train, y_train, epochs=5, batch_size=256)
        else:
            self.model.fit(x_train, y_train)

    def convert_to_onnx(self) -> ModelProto:
        """
        Converts the different modeltypes to ONNX format for interoperability.

        Returns:
            ModelProto: The model in ONNX format.

        Raises:
            ValueError: If the model type is unsupported.
        """
        # Dynamically import necessary modules based on the model type
        if self.modeltype == ModelType.NeuralNetwork:
            return self._convert_neural_network_to_onnx()
        elif self.modeltype in [ModelType.RandomForest, ModelType.GBC]:
            return self._convert_sklearn_to_onnx()
        elif self.modeltype == ModelType.XGBoost:
            return self._convert_xgboost_to_onnx()
        else:
            raise ValueError("Unsupported model type")

    def get_model(self) -> object:
        return self.model

    # Private methods for initialization
    def _init_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    def _init_neural_network(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Dense(64, input_dim=30, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def _init_xgboost(self):
        from xgboost import XGBClassifier
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def _init_gbc(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42, verbose=1)

    # Private methods for ONNX conversion
    def _convert_neural_network_to_onnx(self):
        import tf2onnx
        input_signature = [TensorSpec(shape=[None, 30], dtype=float32, name="X")]
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=input_signature)
        return onnx_model

    def _convert_sklearn_to_onnx(self):
        from skl2onnx import convert_sklearn
        initial_types = [("X", FloatTensorType([None, 30]))]
        return convert_sklearn(self.model, initial_types=initial_types)

    def _convert_xgboost_to_onnx(self):
        from onnxmltools import convert_xgboost
        return convert_xgboost(self.model, initial_types=[("X", FloatTensorType([None, 30]))])
    
class DataProcessor:
    """
    A class that processes data for fraud detection pipeline.

    Attributes:
        data_path (str): The path to the data file.
        x_train (numpy.ndarray): The training data features.
        x_test (numpy.ndarray): The testing data features.
        y_train (numpy.ndarray): The training data labels.
        y_test (numpy.ndarray): The testing data labels.
        test_split (float): The proportion of data to be used for testing.
        scaler (StandardScaler): The scaler used for data preprocessing.

    Methods:
        process_data: Loads, splits, and preprocesses the data.
        load_data: Loads the data from the specified file path.
        split_data: Splits the data into training and testing sets.
        preprocess_data: Preprocesses the data using a scaler.
    """

    def __init__(self, data_path: str, test_split: float = 0.2) -> None:
        self.data_path = data_path
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.test_split = test_split
        self.scaler = StandardScaler()

    def process_data(self) -> tuple:
        """
        Loads, splits, and preprocesses the data.

        Returns:
            tuple: A tuple containing the training and testing data features and labels.
        """
        self.load_data()
        self.split_data()
        self.preprocess_data()
        return self.x_train, self.x_test, self.y_train, self.y_test

    def load_data(self) -> None:
        """
        Loads the data from the specified file path.
        """
        logging.info("Loading data...")
        data = pd.read_csv(self.data_path)
        self.X = data.drop("Class", axis=1)
        self.y = data["Class"]
        logging.info("Data loaded.")

    def split_data(self) -> None:
        """
        Splits the data into training and testing sets.
        """
        logging.info("Splitting data...")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_split, random_state=42
        )
        logging.info("Data split.")

    def preprocess_data(self) -> None:
        """
        Preprocesses the data using a scaler.
        """
        logging.info("Preprocessing data...")
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        logging.info("Data preprocessed.")


class ModelTrainer:
    """
    A class for training machine learning models.

    Args:
        x_train (array-like): The input features for training.
        y_train (array-like): The target variable for training.

    Attributes:
        model: The trained machine learning model.
        x_train (array-like): The input features for training.
        y_train (array-like): The target variable for training.

    Methods:
        train_model: Trains the machine learning model.
    """

    def __init__(self, x_train, y_train, modeltype: ModelType) -> None:
        self.model = None
        self.x_train = x_train
        self.y_train = y_train
        self.modeltype = modeltype

    def train_model(self) -> Model:
        """
        Trains the machine learning model.

        Args:
            n_estimators (int, optional): The number of trees in the random forest. Defaults to 50.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            RandomForestClassifier: The trained random forest classifier model.
        """
        logging.info("Training model, this may take a while...")
        self.model = Model(self.modeltype)
        self.model.train_model(self.x_train, self.y_train)
        logging.info("Model trained.")
        return self.model


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


class ModelManager:
    """
    A class that manages the saving, loading, and prediction of a machine learning model.

    Attributes:
        model_path (str): The path to save/load the model.
        model (InferenceSession): The loaded model.

    Methods:
        save_model: Saves the model as an ONNX file.
        load_model: Loads the model from the saved ONNX file.
        get_prediction: Makes a prediction using the loaded model.
        get_model: Returns the loaded model.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes a new instance of the ModelManager class.

        Args:
            model_path (str): The path to save/load the model.
        """
        self.model_path = model_path
        self.model = None

    def save_model(self, model: Model) -> bool:
        """
        Saves the model as an ONNX file.

        Args:
            model: The trained machine learning model.
            x_train: The input data used for training the model.

        Returns:
            bool: True if the model is successfully saved, False otherwise.
        """
        logging.info("Saving model...")
        try:
            onnx_model = model.convert_to_onnx()
        except Exception as e:
            logging.exception(e)
            return False
        try:
            with open(self.model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except Exception as e:
            logging.exception(e)
            return False
        logging.info("Model saved.")
        return True

    def load_model(self) -> bool:
        """
        Loads the model from the saved ONNX file.

        Returns:
            bool: True if the model is successfully loaded, False otherwise.
        """
        logging.info("Loading model...")
        try:
            with open(self.model_path, "rb") as f:
                self.model = InferenceSession(f.read())
        except Exception as e:
            logging.exception(e)
            logging.info("Model not found.")
            return False
        logging.info("Model loaded.")
        return True

    def get_prediction(self, data) -> int:
        """
        Makes a prediction using the loaded model.

        Args:
            data: The input data for making the prediction.

        Returns:
            int: The predicted value.
        """
        logging.info("Predicting...")
        try:
            # Ensure the input data is of type float32
            data = data.astype(np.float32)
            output = self.model.run(None, {"X": data})[0]
            prediction = np.where(output > 0.5, 1, 0).astype(int)
        except Exception as e:
            logging.exception(e)
            return None
        logging.info(f"Prediction: {prediction}")
        return prediction

    def get_model(self) -> InferenceSession:
        """
        Returns the loaded model.

        Returns:
            InferenceSession: The loaded model.
        """
        return self.model
