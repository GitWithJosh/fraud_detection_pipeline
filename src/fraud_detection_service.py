import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline.model_manager import ModelManager
from pipeline.data_processor import DataProcessor
from pipeline.model_trainer import ModelTrainer
from pipeline.model_evaluator import ModelEvaluator
from pipeline.models import ModelType

class CreditCard:
    """
    A class to represent a credit card with transactions.

    Attributes:
        transactions (list): A list of transactions.
    """

    def __init__(self) -> None:
        self.transactions = []

    def add_transaction(self, transaction: dict) -> None:
        self.transactions.append(transaction)

    def get_transactions(self) -> list:
        return self.transactions


class FraudDetectionService:
    """
    A class to represent a fraud detection service.
    This class serves as an interface to the fraud detection pipeline.

    Attributes:
        model_manager (ModelManager): An instance of the ModelManager class.
        data_processor (DataProcessor): An instance of the DataProcessor class.
        model_evaluator (ModelEvaluator): An instance of the ModelEvaluator class.
        credit_card (CreditCard): An instance of the CreditCard class.
    """

    def __init__(self, model_path: str, modeltype: ModelType) -> None:
        self.model_path = model_path
        self.model_manager = ModelManager(self.model_path)
        data_path = os.path.join(os.getcwd(), "datasets", "creditcard_2023.csv")
        self.data_processor = DataProcessor(data_path, test_split=0.2)
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test
        ) = self.data_processor.process_data()

        self.model_trainer = ModelTrainer(self.x_train, self.y_train, modeltype)
        self.load_model()
        self.model_evaluator = ModelEvaluator(self.model, self.x_test, self.y_test)
        self.credit_card = CreditCard()

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            logging.info("Model not found. Training a new model...")
            self.model = self.model_trainer.train_model()
            self.model_manager.save_model(self.model, self.x_train)
            logging.info("Model trained and loaded successfully!")
        else:
            self.model_manager.load_model()
            self.model = self.model_manager.get_model()
            logging.info("Model loaded successfully!")

    def evaluate_model(self) -> None:
        self.model_evaluator.evaluate_model()

    def get_model_metrics(self) -> dict:
        return self.model_evaluator.get_evaluation_metrics()
    
    def visualize_confusion_matrix(self, figsize=(8, 6)):
        confusion_array = self.model_evaluator.visualize_confusion_matrix()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(confusion_array, annot=True, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        return fig

    def add_transaction_and_evaluate(self, transaction_data: dict) -> int:
        transaction_df = pd.DataFrame([transaction_data])
        self.credit_card.add_transaction(transaction_df.iloc[0])

        # Prepare input data for prediction
        input_data = self.data_processor.scaler.transform(transaction_df.drop(columns=['Class']))

        # Predict fraud for the added transaction
        prediction = self.get_prediction(input_data)
        return prediction

    def get_prediction(self, data: np.ndarray) -> int:
        try:
            prediction = self.model_manager.get_prediction(data.reshape(1, -1))[0]
            return prediction
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return -1

if __name__ == "__main__":
    # Test the FraudDetectionService class
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    fds = FraudDetectionService("model.onnx")
    fds.evaluate_model()

    # Example transaction data to add // This transaction is not fraudulent
    transaction_data_1 = {
        'id': 0,
        'V1': -0.26064780489439815,
        'V2': -0.46964845005363426,
        'V3': 2.4962660826315637,
        'V4': -0.08372391267814633,
        'V5': 0.1296812361545678,
        'V6': 0.7328982498449426,
        'V7': 0.5190136179018007,
        'V8': -0.13000604758867731,
        'V9': 0.7271592691096374,
        'V10': 0.6377345411881967,
        'V11': -0.9870200098799841,
        'V12': 0.2934381004820159,
        'V13': -0.9413861250929441,
        'V14': 0.5490198936308889,
        'V15': 1.8048785784684263,
        'V16': 0.2155979938725388,
        'V17': 0.5123066605849524,
        'V18': 0.3336437173298195,
        'V19': 0.12427015635408474,
        'V20': 0.09120189881650709,
        'V21': -0.11055167961012961,
        'V22': 0.21760614382950005,
        'V23': -0.1347944948772723,
        'V24': 0.1659591154312752,
        'V25': 0.1262799761446219,
        'V26': -0.43482398075374323,
        'V27': -0.08123010860166756,
        'V28': -0.1510454864555771,
        'Amount': 17982.1,
        'Class': 0
    }

    # Example transaction data to add // This transaction is fraudulent
    transaction_data_2 = {
        'id': 1,
        'V1': -1.5141004664822902,
        'V2': 3.283588489906258,
        'V3': -4.880687449627004,
        'V4': 2.3338941804119397,
        'V5': -0.9005140535424711,
        'V6': -2.549100086561628,
        'V7': -2.295246454494773,
        'V8': 0.2614963387133994,
        'V9': -0.23520752694842347,
        'V10': -4.476674738213551,
        'V11': 4.577953553510946,
        'V12': -5.021187860997949,
        'V13': -0.6503562677332273,
        'V14': -5.601598029604834,
        'V15': -0.8711865984072833,
        'V16': -2.1229109023705733,
        'V17': -4.198591072351821,
        'V18': -1.7033888164764764,
        'V19': 0.587308888666001,
        'V20': 0.2527329927677151,
        'V21': 0.671743007518611,
        'V22': -0.3591871007038272,
        'V23': 0.24210316586577618,
        'V24': 0.04496298697621776,
        'V25': -0.4910781271048116,
        'V26': -0.21750614384133165,
        'V27': 0.14255274265508742,
        'V28': 0.08314536647217075,
        'Amount': 200.0,
        'Class': 0
    }

    # Add the first transaction and get prediction
    prediction_1 = fds.add_transaction_and_evaluate(transaction_data_1)
    if prediction_1 == 0:
        print("Prediction for the first added transaction: Not fraudulent")
    else:
        print("Prediction for the first added transaction: Fraudulent")

    # Add the second transaction and get prediction
    prediction_2 = fds.add_transaction_and_evaluate(transaction_data_2)
    if prediction_2 == 0:
        print("Prediction for the second added transaction: Not fraudulent")
    else:
        print("Prediction for the second added transaction: Fraudulent")
