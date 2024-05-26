from fraud_detection_pipeline import DataProcessor, ModelTrainer, ModelEvaluator, ModelManager

class FraudDetectionService:
    """
    A class that represents a fraud detection service.

    Attributes:
        model_path (str): The path to the trained model.

    Methods:
        __init__(self, model_path): Initializes the FraudDetectionService object.
        _load_model_(self): Loads the trained model.
        evaluate_model(self): Evaluates the model using test data.
        get_prediction(self, data): Makes a prediction using the trained model.
    """

    def __init__(self, model_path) -> None:
        """
        Initializes the FraudDetectionService object.

        Args:
            model_path (str): The path to the trained model.
        """
        self.model_manager = ModelManager(model_path)
        self.data_processor = DataProcessor("./creditcard_2023.csv", test_split=0.2)
        self.model = self._load_model_()
        (
            self.x_train, 
            self.x_test, 
            self.y_train, 
            self.y_test
        ) = self.data_processor.process_data()
        self.model_trainer = ModelTrainer(self.x_train, self.y_train)
        self.model_evaluator = ModelEvaluator(self.model, self.x_test, self.y_test)
        
    def _load_model_(self) -> None:
        """
        Loads the trained model.

        Returns:
            None
        """
        if not self.model_manager.load_model():
            self.model = self.model_trainer.train_model()
            self.model_manager.save_model(self.model, self.x_train)
        else:
            self.model = self.model_manager.get_model()
        
    def evaluate_model(self) -> None:
        """
        Evaluates the model using test data.

        Returns:
            None
        """
        self.model_evaluator.evaluate_model()
    
    def get_prediction(self, data) -> int:
        """
        Makes a prediction using the trained model.

        Args:
            data: The input data for prediction.

        Returns:
            int: The predicted class label.
        """
        return self.model_manager.get_prediction(data)