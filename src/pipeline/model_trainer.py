import logging

from .models import Model, ModelType, RandomForest, NeuralNetwork, XGBoost, GBC

class ModelTrainer:
    """
    A class for training machine learning models.

    Args:
        x_train (array-like): The input features for training.
        y_train (array-like): The target variable for training.

    Attributes:
        model: The trained machine learning model.
        x_train (array-like): The input features for training.
        y_train (array-like): The target variable for training.

    Methods:
        train_model: Trains the machine learning model.
    """

    def __init__(self, x_train, y_train, modeltype: ModelType) -> None:
        self.model = None
        self.x_train = x_train
        self.y_train = y_train
        self.modeltype = modeltype

    def train_model(self) -> Model:
        """
        Trains the machine learning model.

        Args:
            n_estimators (int, optional): The number of trees in the random forest. Defaults to 50.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            RandomForestClassifier: The trained random forest classifier model.
        """
        logging.info("Training model, this may take a while...")
        if self.modeltype == ModelType.RandomForest:
            self.model = RandomForest()
        elif self.modeltype == ModelType.NeuralNetwork:
            self.model = NeuralNetwork()
        elif self.modeltype == ModelType.XGBoost:
            self.model = XGBoost()
        elif self.modeltype == ModelType.GBC:
            self.model = GBC()
        
        self.model.train_model(self.x_train, self.y_train)
        logging.info("Model trained.")
        return self.model