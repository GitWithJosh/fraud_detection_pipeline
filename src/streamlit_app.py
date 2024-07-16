import logging
from io import BytesIO
import streamlit as st
import pandas as pd
from fraud_detection_service import FraudDetectionService, CreditCard
from pipeline.models import ModelType

class StreamlitLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.log_messages = []
    
    def emit(self, record: logging.LogRecord) -> None:
        self.log_messages.append(self.format(record))

class StreamlitApp:
    def __init__(self):
        """
        StreamlitApp class constructor.
        Initializes the Streamlit app and sets up the session state variables.
        """
        logging.basicConfig(level=logging.INFO)
        self.log_handler = StreamlitLogHandler()
        logging.getLogger().addHandler(self.log_handler)
        
        st.set_page_config(layout="wide", page_title="Fraud Detection Service")
        
        if 'cards' not in st.session_state:
            st.session_state.cards = []
        
        st.title("Fraud Detection Service")
        
        st.subheader("Choose Model to use")
        
        # Initialize session state variables
        if "model" not in st.session_state:
            st.session_state.model = None
            self.fds = None
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "Select Model"
        
        self.selected_model = st.selectbox("Model", ["Select Model", "Random Forest", "XGBoost", "Neural Network", "Gradient Boosting Classifier"])
        
        # Load model if selected model has changed, otherwise use the existing model in session state
        if self.selected_model != st.session_state.selected_model and self.selected_model != "Select Model":
            self.load_model()
            st.session_state.selected_model = self.selected_model
            st.session_state.model = self.fds
        else:
            self.fds = st.session_state.model
            self.selected_model = st.session_state.selected_model
            
        self.display_logs()
        
        if self.selected_model != "Select Model":
            st.button("Add Credit Card", on_click=self.add_credit_card)
            
            for card in reversed(st.session_state.cards):
                self.display_upload_transactions(card)
            
            cm_figure = self.visualize_confusion_matrix()
            self.display_confusion_matrix(cm_figure)
            
            
    def load_model(self):
        """
        Loads the selected model based on the selected_model attribute.
        """
        with st.spinner("Loading model..."):
            if self.selected_model == "Random Forest":
                self.fds = FraudDetectionService("models/randomforest.onnx", modeltype=ModelType.RandomForest)
            elif self.selected_model == "XGBoost":
                self.fds = FraudDetectionService("models/xgboost.onnx", modeltype=ModelType.XGBoost)
            elif self.selected_model == "Neural Network":
                self.fds = FraudDetectionService("models/neuralnetwork.onnx", modeltype=ModelType.NeuralNetwork)
            elif self.selected_model == "Gradient Boosting Classifier":
                self.fds = FraudDetectionService("models/gbc.onnx", modeltype=ModelType.GBC)
            else:
                raise ValueError("Invalid model selected")
    
    def add_credit_card(self):
        """
        Adds a new credit card to the session state list.
        """
        # Append a new CreditCard to the session state list
        st.session_state.cards.append(CreditCard(id=len(st.session_state.cards)))
        st.success(f"Added credit card #{len(st.session_state.cards)}")
        
    def display_upload_transactions(self, card: CreditCard):
        """
        Displays the upload transactions section for a given credit card.

        Args:
            card (CreditCard): The credit card object to display the upload transactions section for.
        """
        st.subheader("Upload a CSV file to predict fraud in transactions")
            
        self.uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=card.id)
    
        if self.uploaded_file is not None:
            with st.spinner("Predicting transactions..."):
                try:
                    transactions_df, predictions = self.predict_transactions(card)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
                st.success("Prediction complete!")
            
            # Display results after processing
            self.display_results(transactions_df, predictions)
        
    def predict_transactions(self, card: CreditCard):    
        """
        Predicts fraud in transactions based on the uploaded CSV file.

        Args:
            card (CreditCard): The credit card object to use for prediction.
        """
        transactions_df = pd.read_csv(self.uploaded_file)
        predictions = []

        for index, row in transactions_df.iterrows():
            transaction_data = row.to_dict()
            prediction = self.fds.add_transaction_and_evaluate(transaction_data, card)
            predictions.append(prediction)
        
        return transactions_df, predictions

    def visualize_confusion_matrix(self):
        """
        Visualizes the confusion matrix for the selected model.
        """
        cm_figure = self.fds.visualize_confusion_matrix(figsize=(6, 4))
        return cm_figure

    def display_results(self, transactions_df, predictions):
        """
        Displays the uploaded transactions and fraud prediction results.

        Args:
            transactions_df (pd.DataFrame): The uploaded transactions DataFrame.
            predictions (List[int]): The list of predictions for each transaction.
        """
        st.subheader("Uploaded Transactions")
        
        with st.expander("Show Transactions", expanded=True):
            st.write(transactions_df)

        st.subheader("Fraud Prediction Results")
        with st.expander("Show Predictions", expanded=True):
            transaction_table = pd.DataFrame({
            "Transaction ID": [transaction_data.get('id', 'N/A') for transaction_data in transactions_df.to_dict('records')],
            "Prediction": ['Fraudulent' if prediction == 1 else 'Not Fraudulent' for prediction in predictions],
            "Actual": ['Fraudulent' if transaction_data.get('Class', 'Unknown') == 1 else 'Not Fraudulent' for transaction_data in transactions_df.to_dict('records')]
            })
            st.dataframe(transaction_table)

    def display_confusion_matrix(self, cm_figure):
        """
        Displays the confusion matrix for the selected model.

        Args:
            cm_figure (matplotlib.figure.Figure): The confusion matrix figure to display.
        """
        # Display confusion matrix in dropdown expander
        st.subheader("Model performance")
        with st.expander("Show Model Performance"):
            st.write("Model performance metrics")
            for metric, value in self.fds.get_model_metrics().items():
                st.write(f"{metric}: {value}")
            st.write("Confusion Matrix")
            buf = BytesIO()
            cm_figure.savefig(buf, format="png")
            st.image(buf)
        
    def display_logs(self):
        """
        Displays the logs recorded by the StreamlitLogHandler.
        """
        with st.expander("Show Logs"):
            for message in self.log_handler.log_messages:
                st.text(message)


if __name__ == "__main__":
    app = StreamlitApp()