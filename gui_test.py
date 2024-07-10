import streamlit as st
import pandas as pd
import numpy as np
from fraud_detection_service import FraudDetectionService

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
        fds = FraudDetectionService("model.onnx")
        # Display the uploaded transactions
        st.subheader("Uploaded Transactions")
        st.write(transactions_df)

        st.subheader("Fraud Prediction Results")

        # Iterate through each transaction in the CSV and predict fraud
        for index, row in transactions_df.iterrows():
            transaction_data = row.to_dict()
            prediction = fds.add_transaction_and_evaluate(transaction_data)
            actual = transaction_data.get('Class', 'Unknown')

            st.write(f"Transaction ID: {transaction_data.get('id', 'N/A')} | Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'} | Actual: {'Fraudulent' if actual == 1 else 'Not Fraudulent'}")

        # Visualize confusion matrix
        # st.subheader("Confusion Matrix")
        # cm = evaluator.visualize_confusion_matrix()
        # st.write(cm)

if __name__ == "__main__":
    main()
