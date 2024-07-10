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

class ModelType(Enum):
    RandomForest = 1
    NeuralNetwork = 2
    XGBoost = 3
    GBC = 4

class Model:
    def __init__(self, modeltype: ModelType):
        self.modeltype = modeltype
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.modeltype == ModelType.RandomForest:
            
            # Only import modules if the model type is RandomForest
            from sklearn.ensemble import RandomForestClassifier
                        
            return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        elif self.modeltype == ModelType.NeuralNetwork:
            
            # Only import modules if the model type is NeuralNetwork
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            
            model = Sequential()
            model.add(Dense(64, input_dim=30, activation='relu'))
            model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
            model.add(Dense(32, activation='relu'))  # Additional hidden layer
            model.add(Dropout(0.2))  # Another Dropout layer
            model.add(Dense(1, activation='sigmoid'))
            model.output_names = ['output']
            optimizer = Adam(learning_rate=0.001)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return model
        elif self.modeltype == ModelType.XGBoost:
            
            # Only import modules if the model type is XGBoost
            from xgboost import XGBClassifier
            
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif self.modeltype == ModelType.GBC:
            
            # Only import modules if the model type is GBC
            from sklearn.ensemble import GradientBoostingClassifier
            
            return GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42, verbose=1)
        else:
            raise ValueError("Unsupported model type")
        
    def get_model(self):
        return self.model
    
    def train_model(self, x_train, y_train):
        if self.modeltype == ModelType.NeuralNetwork:
            self.model.fit(x_train, y_train, epochs=5, batch_size=256)
        else:
            self.model.fit(x_train, y_train)
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

    def save_model(self, model: Model, x_train) -> bool:
        """
        Saves the model as an ONNX file.

        Args:
            model: The trained machine learning model.
            x_train: The input data used for training the model.

        Returns:
            bool: True if the model is successfully saved, False otherwise.
        """
        # Import necessary modules based on the model type
        from skl2onnx import convert_sklearn
        from onnxmltools import convert_xgboost
        import tf2onnx
        from tensorflow import TensorSpec, float32
        from skl2onnx.common.data_types import FloatTensorType
        
        logging.info("Saving model...")
        try:
            if model.modeltype == ModelType.NeuralNetwork:
                input_signature = [TensorSpec(shape=[None, 30], dtype=float32, name="X")]
                onnx_model, _ = tf2onnx.convert.from_keras(model.get_model(), input_signature=input_signature)
            elif model.modeltype == ModelType.RandomForest or model.modeltype == ModelType.GBC:
                initial_types = [("X", FloatTensorType([None, 30]))]
                onnx_model = convert_sklearn(model.get_model(), initial_types=initial_types)
            elif model.modeltype == ModelType.XGBoost:
                onnx_model = convert_xgboost(model.get_model(), initial_types=[("X", FloatTensorType([None, 30]))])
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


def main():
    """
    The main function that demonstrates the fraud detection pipeline.
    """
    # Example usage
    data_processor = DataProcessor("./creditcard_2023.csv", test_split=0.2)
    x_train, x_test, y_train, y_test = data_processor.process_data()
    model_manager = ModelManager("./model.onnx")

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
    # Visualize the confusion matrix
    cm = evaluator.visualize_confusion_matrix()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()


    print("Predictions:")
    pred1 = model_manager.get_prediction(x_test[0:1])[0]
    actual = y_test.iloc[0]
    print(f"Prediction 1: {pred1} Actual: {actual}")
    pred2 = model_manager.get_prediction(x_test[1:2])[0]
    actual2 = y_test.iloc[1]
    print(f"Prediction 2: {pred2} Actual: {actual2}")
    pred3 = model_manager.get_prediction(x_test[2:3])[0]
    actual3 = y_test.iloc[2]
    print(f"Prediction 3: {pred3} Actual: {actual3}")


if __name__ == "__main__":
    """
    If __name__ == "__main__": is used to check whether the current script is run as the main program.
    If the script is imported as a module in another script, the block of code under if __name__ == "__main__": will not run.
    If the script is run directly, the block of code under if __name__ == "__main__": will run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        main()
    except Exception as e:
        logging.exception(e)
