from io import BytesIO, StringIO
import streamlit as st
import pandas as pd
from fraud_detection_service import FraudDetectionService
from fraud_detection_pipeline import ModelType
import logging

class StreamlitLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.log_messages = []
    
    def emit(self, record: logging.LogRecord) -> None:
        self.log_messages.append(self.format(record))

class StreamlitApp:
    def __init__(self):
        
        logging.basicConfig(level=logging.INFO)
        log_stream = StringIO
        self.log_handler = StreamlitLogHandler()
        logging.getLogger().addHandler(self.log_handler)
        
        st.set_page_config(layout="wide")
        st.title("Fraud Detection Service")
        
        st.subheader("Choose Model to use")
        
        self.selected_model = st.selectbox("Model", ["Select Model", "Random Forest", "XGBoost", "Neural Network", "Gradient Boosting Classifier"])
        
        self.display_logs()
        with st.spinner("Loading model..."):
            if self.selected_model == "Random Forest":
                self.fds = FraudDetectionService("models/randomforest.onnx", modeltype=ModelType.RandomForest)
            elif self.selected_model == "XGBoost":
                self.fds = FraudDetectionService("models/xgboost.onnx", modeltype=ModelType.XGBoost)
            elif self.selected_model == "Neural Network":
                self.fds = FraudDetectionService("models/neuralnetwork.onnx", modeltype=ModelType.NeuralNetwork)
            elif self.selected_model == "Gradient Boosting Classifier":
                self.fds = FraudDetectionService("models/gbc.onnx", modeltype=ModelType.GBC)
        
        if self.selected_model != "Select Model":
            st.subheader("Upload a CSV file to predict fraud in transactions")
            
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
        
    def display_logs(self):
        with st.expander("Show Logs"):
            for message in self.log_handler.log_messages:
                st.text(message)
            

if __name__ == "__main__":
    app = StreamlitApp()
    app.display_logs()