import logging
from io import BytesIO
import streamlit as st
import pandas as pd
from fraud_detection_service import FraudDetectionService, CreditCard
from pipeline.models import ModelType

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []

    def emit(self, record: logging.LogRecord):
        self.log_messages.append(self.format(record))

class StreamlitApp:
    def __init__(self):
        """
        StreamlitApp class constructor.
        Initializes the Streamlit app and sets up the session state variables.
        """
        self.setup_logging()
        self.setup_streamlit()

        self.selected_model = self.model_selection()
        self.handle_model_selection()
        self.display_logs()

        if st.session_state.selected_model_main != "Select Model" and self.fds:
            st.button("Add Credit Card", on_click=self.add_credit_card)
            for card in reversed(st.session_state.cards):
                self.display_upload_transactions(card)
            
            cm_figure = self.visualize_confusion_matrix()
            self.display_confusion_matrix(cm_figure)
            
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.log_handler = StreamlitLogHandler()
        logging.getLogger().addHandler(self.log_handler)

    def setup_streamlit(self):
        st.set_page_config(layout="wide", page_title="Fraud Detection Service")
        st.title("Fraud Detection Service")
        st.subheader("Choose Model to use")

        if 'cards' not in st.session_state:
            st.session_state.cards = []
        if "page" not in st.session_state:
            st.session_state.page = "main"
        if "model" not in st.session_state:
            st.session_state.model = None
            self.fds = None
        if "selected_model_main" not in st.session_state:
            st.session_state.selected_model_main = "Select Model"
        if 'model_names' not in st.session_state:
            st.session_state.model_names = {}

    def model_selection(self):
        available_models = ["Select Model", "Random Forest", "XGBoost", "Neural Network", "Gradient Boosting Classifier"]
        available_models += list(st.session_state.model_names.keys())
        return st.selectbox("Model", available_models, key="main_page")

    def handle_model_selection(self):
        if st.session_state.page == "train":
            st.session_state.page = "main"
            st.session_state.selected_model_main = "Select Model"
            st.session_state.model = None
            self.fds = None

        if self.selected_model != st.session_state.selected_model_main and self.selected_model != "Select Model":
            self.load_model()
            st.session_state.selected_model_main = self.selected_model
            st.session_state.model = self.fds
        else:
            self.fds = st.session_state.model
            self.selected_model = st.session_state.selected_model_main

    def load_model(self):
        with st.spinner("Loading model..."):
            model_paths = {
                "Random Forest": ("models/randomforest.onnx", ModelType.RandomForest),
                "XGBoost": ("models/xgboost.onnx", ModelType.XGBoost),
                "Neural Network": ("models/neuralnetwork.onnx", ModelType.NeuralNetwork),
                "Gradient Boosting Classifier": ("models/gbc.onnx", ModelType.GBC)
            }
            if self.selected_model in model_paths:
                path, model_type = model_paths[self.selected_model]
                self.fds = FraudDetectionService(path, modeltype=model_type)
            else:
                path = f"models/{self.selected_model}.onnx"
                self.fds = FraudDetectionService(path, modeltype=st.session_state.model_names[self.selected_model])

    def add_credit_card(self):
        st.session_state.cards.append(CreditCard(id=len(st.session_state.cards)))
        st.success(f"Added credit card #{len(st.session_state.cards)}")

    def display_upload_transactions(self, card):
        st.subheader("Upload a CSV file to predict fraud in transactions")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=card.id)

        if uploaded_file:
            with st.spinner("Predicting transactions..."):
                try:
                    transactions_df, predictions = self.predict_transactions(uploaded_file, card)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
                st.success("Prediction complete!")
            self.display_results(transactions_df, predictions)

    def predict_transactions(self, uploaded_file, card):
        transactions_df = pd.read_csv(uploaded_file)
        predictions = [self.fds.add_transaction_and_evaluate(row.to_dict(), card) for _, row in transactions_df.iterrows()]
        return transactions_df, predictions

    def visualize_confusion_matrix(self):
        return self.fds.visualize_confusion_matrix(figsize=(6, 4))

    def display_results(self, transactions_df, predictions):
        st.subheader("Uploaded Transactions")
        with st.expander("Show Transactions", expanded=True):
            st.write(transactions_df)

        st.subheader("Fraud Prediction Results")
        with st.expander("Show Predictions", expanded=True):
            transaction_table = pd.DataFrame({
                "Transaction ID": transactions_df.get('id', 'N/A'),
                "Prediction": ['Fraudulent' if p == 1 else 'Not Fraudulent' for p in predictions],
                "Actual": ['Fraudulent' if x == 1 else 'Not Fraudulent' for x in transactions_df.get('Class', 'Unknown')]
            })
            st.dataframe(transaction_table)

    def display_confusion_matrix(self, cm_figure):
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
        with st.expander("Show Logs"):
            for message in self.log_handler.log_messages:
                st.text(message)

if __name__ == "__main__":
    app = StreamlitApp()
