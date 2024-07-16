import logging
import os

import streamlit as st
import pandas as pd

from fraud_detection_service import FraudDetectionService
from pipeline.models import ModelType

class StreamlitLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.log_messages = []
    
    def emit(self, record: logging.LogRecord) -> None:
        self.log_messages.append(self.format(record))

class StreamlitApp:
    def __init__(self):
        
        logging.basicConfig(level=logging.INFO)
        self.log_handler = StreamlitLogHandler()
        logging.getLogger().addHandler(self.log_handler)
        
        st.set_page_config(layout="wide", page_title="Fraud Detection Service")
        
        st.title("Train New Model")
        
        st.subheader("Choose Model to use")
        
        # Initialize session state variables
        if "page" not in st.session_state:
            st.session_state.page = "train"
        if "selected_model_train" not in st.session_state:
            st.session_state.selected_model_train = "Select Model"
        if 'model_names' not in st.session_state:
            st.session_state.model_names = {}      
        
        self.selected_model = st.selectbox("Model", ["Select Model", "Random Forest", "XGBoost", "Neural Network", "Gradient Boosting Classifier"]
                                           , key="train_page")
        
        # Upload Training Data
        self.uploaded_file = st.file_uploader("Upload Training Data", type=["csv"])
        
        if st.session_state.page == "main":
            st.session_state.page = "train"
            st.session_state.selected_model_train = "Select Model"
        
        if self.uploaded_file is not None:
            self.data_path = os.path.join(os.getcwd(), "datasets", self.uploaded_file.name)
            with open(self.data_path, "wb") as f:
                f.write(self.uploaded_file.getvalue())
        
        if self.uploaded_file is not None:
            self.model_name = st.text_input("Give your model a name")
            if self.model_name:
                self.train_model()

            
        self.display_logs()
        
    def train_model(self):
        # Load model if selected model has changed, otherwise use the existing model in session state
        if self.selected_model != st.session_state.selected_model_train:
            self.load_model()
        else:
            self.selected_model = st.session_state.selected_model_train
    
    def load_model(self):
        with st.spinner("Training model..."):
            if self.selected_model == "Random Forest":
                self.fds = FraudDetectionService(f"models/{self.model_name}.onnx", modeltype=ModelType.RandomForest, data_path=self.data_path)
                st.session_state.model_names[self.model_name] = ModelType.RandomForest
            elif self.selected_model == "XGBoost":
                self.fds = FraudDetectionService(f"models/{self.model_name}.onnx", modeltype=ModelType.XGBoost, data_path=self.data_path)
                st.session_state.model_names[self.model_name] = ModelType.XGBoost
            elif self.selected_model == "Neural Network":
                self.fds = FraudDetectionService(f"models/{self.model_name}.onnx", modeltype=ModelType.NeuralNetwork, data_path=self.data_path)
                st.session_state.model_names[self.model_name] = ModelType.NeuralNetwork
            elif self.selected_model == "Gradient Boosting Classifier":
                self.fds = FraudDetectionService(f"models/{self.model_name}.onnx", modeltype=ModelType.GBC, data_path=self.data_path)
                st.session_state.model_names[self.model_name] = ModelType.GBC
            else:
                raise ValueError("Invalid model selected")
        st.success("New model trained successfully")
            
    def display_logs(self):
        with st.expander("Show Logs"):
            for message in self.log_handler.log_messages:
                st.text(message)
            
if __name__ == "__main__":
    app = StreamlitApp()