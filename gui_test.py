import streamlit as st
import pandas as pd
import numpy as np
from fraud_detection_pipeline import (
    DataProcessor, ModelTrainer, ModelEvaluator, ModelManager
)
import logging

class CreditCard:
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

    def __init__(self, model_path: str) -> None:
        self.model_manager = ModelManager(model_path)
        self.data_processor = DataProcessor("./creditcard_2023.csv", test_split=0.2)
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test
        ) = self.data_processor.process_data()

        self.model_evaluator = None
        self.credit_card = CreditCard()

        self.load_model()

    def load_model(self) -> None:
        if not self.model_manager.load_model():
            model_trainer = ModelTrainer(self.x_train, self.y_train)
            self.model = model_trainer.train_model()
            self.model_manager.save_model(self.model, self.x_train)
        else:
            self.model = self.model_manager.get_model()

        self.model_evaluator = ModelEvaluator(self.model, self.x_test, self.y_test)

    def evaluate_model(self) -> None:
        self.model_evaluator.evaluate_model()

    def add_transaction_and_evaluate(self, transaction_data: dict) -> int:
        transaction_df = pd.DataFrame([transaction_data])
        self.credit_card.add_transaction(transaction_df.iloc[0])

        # Prepare input data for prediction
        input_data = self.data_processor.scaler.transform(transaction_df.drop(columns=['Class']))

        # Predict fraud for the added transaction
        prediction = self.get_prediction(input_data)
        return prediction

    def get_prediction(self, data: np.ndarray) -> int:
        # Ensure input data is in the expected format
        if isinstance(data, np.ndarray):
            try:
                prediction = self.model_manager.get_prediction(data.reshape(1, -1))[0]
                return prediction
            except Exception as e:
                logging.error(f"Error during prediction: {str(e)}")
                return -1
        else:
            raise ValueError("Input data must be provided as a numpy array.")


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Fraud Detection Service")
    st.subheader("Upload a CSV file to predict fraud in transactions")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the uploaded CSV file
        transactions_df = pd.read_csv(uploaded_file)

        # Initialize the fraud detection service
        model_path = "model.onnx"
        data_processor = DataProcessor("./creditcard_2023.csv", test_split=0.2)
        x_train, x_test, y_train, y_test = data_processor.process_data()
        model_manager = ModelManager(model_path)

        if not model_manager.load_model():
            # Train model if not already loaded
            # (Your training code here)
            pass
        else:
            model = model_manager.get_model()

        evaluator = ModelEvaluator(model, x_test, y_test)
        evaluator.evaluate_model()

        # Display the uploaded transactions
        st.subheader("Uploaded Transactions")
        st.write(transactions_df)

        st.subheader("Fraud Prediction Results")

        # Iterate through each transaction in the CSV and predict fraud
        for index, row in transactions_df.iterrows():
            transaction_data = row.to_dict()
            prediction = model_manager.get_prediction(x_test[index:index+1])[0]
            actual = transaction_data.get('Class', 'Unknown')

            st.write(f"Transaction ID: {transaction_data.get('id', 'N/A')} | Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'} | Actual: {'Fraudulent' if actual == 1 else 'Not Fraudulent'}")

        # Visualize confusion matrix
        # st.subheader("Confusion Matrix")
        # cm = evaluator.visualize_confusion_matrix()
        # st.write(cm)

if __name__ == "__main__":
    main()
