import logging

import numpy as np
from onnxruntime import InferenceSession

from .models import Model


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