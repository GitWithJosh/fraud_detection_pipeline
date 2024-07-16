import logging
import os
import streamlit as st
from fraud_detection_service import FraudDetectionService
from pipeline.models import ModelType

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []

    def emit(self, record: logging.LogRecord):
        self.log_messages.append(self.format(record))

class StreamlitApp:
    def __init__(self):
        self.setup_logging()
        self.setup_streamlit()

        self.selected_model = self.model_selection()
        self.uploaded_file = self.upload_training_data()
        self.handle_training()

        self.display_logs()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.log_handler = StreamlitLogHandler()
        logging.getLogger().addHandler(self.log_handler)

    def setup_streamlit(self):
        st.set_page_config(layout="wide", page_title="Fraud Detection Service")
        st.title("Train New Model")
        st.subheader("Choose Model to use")

        if "page" not in st.session_state:
            st.session_state.page = "train"
        if "selected_model_train" not in st.session_state:
            st.session_state.selected_model_train = "Select Model"
        if 'model_names' not in st.session_state:
            st.session_state.model_names = {}

    def model_selection(self):
        return st.selectbox(
            "Model", 
            ["Select Model", "Random Forest", "XGBoost", "Neural Network", "Gradient Boosting Classifier"], 
            key="train_page"
        )

    def upload_training_data(self):
        uploaded_file = st.file_uploader("Upload Training Data", type=["csv"])
        if uploaded_file:
            data_path = os.path.join(os.getcwd(), "datasets", uploaded_file.name)
            with open(data_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            return data_path
        return None

    def handle_training(self):
        if st.session_state.page == "main":
            st.session_state.page = "train"
            st.session_state.selected_model_train = "Select Model"

        if self.uploaded_file:
            model_name = st.text_input("Give your model a name")
            if model_name:
                self.train_model(model_name)

    def train_model(self, model_name):
        if self.selected_model != st.session_state.selected_model_train:
            self.load_model(model_name)
        else:
            self.selected_model = st.session_state.selected_model_train

    def load_model(self, model_name):
        model_paths = {
            "Random Forest": ModelType.RandomForest,
            "XGBoost": ModelType.XGBoost,
            "Neural Network": ModelType.NeuralNetwork,
            "Gradient Boosting Classifier": ModelType.GBC
        }

        with st.spinner("Training model..."):
            if self.selected_model in model_paths:
                model_type = model_paths[self.selected_model]
                FraudDetectionService(f"models/{model_name}.onnx", modeltype=model_type, data_path=self.uploaded_file)
                st.session_state.model_names[model_name] = model_type
                st.success("New model trained successfully")
            else:
                st.error("Invalid model selected")

    def display_logs(self):
        with st.expander("Show Logs"):
            for message in self.log_handler.log_messages:
                st.text(message)

if __name__ == "__main__":
    StreamlitApp()