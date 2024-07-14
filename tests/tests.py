import sys
import os

import unittest
from unittest.mock import patch, ANY
from pandas import DataFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.streamlit_app import StreamlitApp, StreamlitLogHandler
from src.pipeline.models import ModelType

class TestStreamlitApp(unittest.TestCase):

    def setUp(self):
        self.app = StreamlitApp()
        self.app.log_handler = StreamlitLogHandler()
        self.app.log_handler.log_messages = ["Test log message"]

    @patch('src.streamlit_app.st')
    def test_display_logs(self, mock_st):
        self.app.display_logs()
        mock_st.expander.assert_called_once_with("Show Logs")
        mock_st.text.assert_called_once_with("Test log message")

    @patch('src.streamlit_app.st')
    def test_display_results(self, mock_st):
        transactions_df = DataFrame({'id': [1], 'Class': [0]})
        predictions = [0]
        self.app.display_results(transactions_df, predictions)
        mock_st.subheader.assert_any_call("Uploaded Transactions")
        mock_st.write.assert_called_with(ANY)
        mock_st.subheader.assert_any_call("Fraud Prediction Results")

    @patch('src.streamlit_app.st')
    @patch('src.streamlit_app.FraudDetectionService')
    def test_load_model(self, mock_fds, mock_st):
        self.app.selected_model = "Random Forest"
        self.app.load_model()
        mock_fds.assert_called_once_with("models/randomforest.onnx", modeltype=ModelType.RandomForest)
        mock_st.spinner.assert_called_once_with("Loading model...")

if __name__ == '__main__':
    unittest.main()