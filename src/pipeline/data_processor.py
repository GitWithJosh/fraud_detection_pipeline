import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    A class that processes data for fraud detection pipeline.

    Attributes:
        data_path (str): The path to the data file.
        x_train (numpy.ndarray): The training data features.
        x_test (numpy.ndarray): The testing data features.
        y_train (numpy.ndarray): The training data labels.
        y_test (numpy.ndarray): The testing data labels.
        test_split (float): The proportion of data to be used for testing.
        scaler (StandardScaler): The scaler used for data preprocessing.

    Methods:
        process_data: Loads, splits, and preprocesses the data.
        load_data: Loads the data from the specified file path.
        split_data: Splits the data into training and testing sets.
        preprocess_data: Preprocesses the data using a scaler.
    """

    def __init__(self, data_path: str, test_split: float = 0.2) -> None:
        self.data_path = data_path
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.test_split = test_split
        self.scaler = StandardScaler()

    def process_data(self) -> tuple:
        """
        Loads, splits, and preprocesses the data.

        Returns:
            tuple: A tuple containing the training and testing data features and labels.
        """
        self.load_data()
        self.split_data()
        self.preprocess_data()
        return self.x_train, self.x_test, self.y_train, self.y_test

    def load_data(self) -> None:
        """
        Loads the data from the specified file path.
        """
        logging.info("Loading data...")
        data = pd.read_csv(self.data_path)
        self.X = data.drop("Class", axis=1)
        self.y = data["Class"]
        logging.info("Data loaded.")

    def split_data(self) -> None:
        """
        Splits the data into training and testing sets.
        """
        logging.info("Splitting data...")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_split, random_state=42
        )
        logging.info("Data split.")

    def preprocess_data(self) -> None:
        """
        Preprocesses the data using a scaler.
        """
        logging.info("Preprocessing data...")
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        logging.info("Data preprocessed.")