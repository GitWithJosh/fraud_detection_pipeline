from io import BytesIO
import time
import streamlit as st
import pandas as pd
import numpy as np
from fraud_detection_service import FraudDetectionService

class StreamlitApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Fraud Detection Service")
        
        st.subheader("Upload a CSV file to predict fraud in transactions")

        self.fds = FraudDetectionService("model.onnx")
        
        self.uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if self.uploaded_file is not None:
            with st.spinner("Predicting transactions..."):
                try:
                    transactions_df, predictions = self.predict_transactions()
                    cm_figure = self.visualize_confusion_matrix()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
                st.success("Prediction complete!")
            
            # Display results after processing
            self.display_results(transactions_df, predictions)
            self.display_confusion_matrix(cm_figure)
            
    def predict_transactions(self):
        transactions_df = pd.read_csv(self.uploaded_file)
        predictions = []

        for index, row in transactions_df.iterrows():
            transaction_data = row.to_dict()
            prediction = self.fds.add_transaction_and_evaluate(transaction_data)
            predictions.append(prediction)
        
        return transactions_df, predictions

    def visualize_confusion_matrix(self):
        cm_figure = self.fds.visualize_confusion_matrix(figsize=(6, 4))
        return cm_figure

    def display_results(self, transactions_df, predictions):
        st.subheader("Uploaded Transactions")
        st.write(transactions_df)

        st.subheader("Fraud Prediction Results")
        for transaction_data, prediction in zip(transactions_df.to_dict('records'), predictions):
            actual = transaction_data.get('Class', 'Unknown')
            st.write(f"Transaction ID: {transaction_data.get('id', 'N/A')} | Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'} | Actual: {'Fraudulent' if actual == 1 else 'Not Fraudulent'}")

    def display_confusion_matrix(self, cm_figure):
        st.subheader("Confusion Matrix")
        buf = BytesIO()
        cm_figure.savefig(buf, format="png")
        st.image(buf)

if __name__ == "__main__":
    app = StreamlitApp()
