from enum import Enum

from onnx import ModelProto
from skl2onnx.common.data_types import FloatTensorType

class ModelType(Enum):
    RandomForest = 1
    NeuralNetwork = 2
    XGBoost = 3
    GBC = 4

class Model:
    """
    Represents a machine learning model abstracted from different model types.

    This class encapsulates the model initialization, training, and conversion to ONNX format,
    supporting multiple types of models including RandomForest, NeuralNetwork, XGBoost, and GBC.
    Though additional model types can be added by extending the ModelType enum and implementing the necessary methods.

    Attributes:
        modeltype (ModelType): The type of the model.
        model (object): The instantiated model object.

    Methods:
        initialize_model: Initializes the model based on its type.
        train_model: Trains the model using the provided training data.
        convert_to_onnx: Converts the model to ONNX format for interoperability.
        get_model: Returns the instantiated model object.
    """
    def __init__(self, modeltype: ModelType):
        self.modeltype = modeltype
        self.model = self.initialize_model()

    def initialize_model(self) -> object:
        """
        Initializes the model based on its type.

        Returns:
            object: The instantiated model object.

        Raises:
            ValueError: If the model type is unsupported.
        """
        if self.modeltype == ModelType.RandomForest:
            return self._init_random_forest()
        elif self.modeltype == ModelType.NeuralNetwork:
            return self._init_neural_network()
        elif self.modeltype == ModelType.XGBoost:
            return self._init_xgboost()
        elif self.modeltype == ModelType.GBC:
            return self._init_gbc()
        else:
            raise ValueError("Unsupported model type")

    def train_model(self, x_train, y_train) -> None:
        """
        Trains the model using the provided training data.

        Parameters:
            x_train: The training data features.
            y_train: The training data labels.
        """
        if self.modeltype == ModelType.NeuralNetwork:
            self.model.fit(x_train, y_train, epochs=5, batch_size=256)
        else:
            self.model.fit(x_train, y_train)

    def convert_to_onnx(self) -> ModelProto:
        """
        Converts the different modeltypes to ONNX format for interoperability.

        Returns:
            ModelProto: The model in ONNX format.

        Raises:
            ValueError: If the model type is unsupported.
        """
        # Dynamically import necessary modules based on the model type
        if self.modeltype == ModelType.NeuralNetwork:
            return self._convert_neural_network_to_onnx()
        elif self.modeltype in [ModelType.RandomForest, ModelType.GBC]:
            return self._convert_sklearn_to_onnx()
        elif self.modeltype == ModelType.XGBoost:
            return self._convert_xgboost_to_onnx()
        else:
            raise ValueError("Unsupported model type")

    def get_model(self) -> object:
        return self.model

    # Private methods for initialization
    def _init_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    def _init_neural_network(self):
        # Ignore type checking for TensorFlow imports cause by bug in TensorFlow 2.6.2
        from tensorflow.keras.models import Sequential # type: ignore
        from tensorflow.keras.layers import Dense, Dropout # type: ignore
        from tensorflow.keras.optimizers import Adam # type: ignore

        model = Sequential([
            Dense(64, input_dim=30, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def _init_xgboost(self):
        from xgboost import XGBClassifier
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def _init_gbc(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42, verbose=1)

    # Private methods for ONNX conversion
    def _convert_neural_network_to_onnx(self):
        from tensorflow import TensorSpec, float32
        import tf2onnx
        input_signature = [TensorSpec(shape=[None, 30], dtype=float32, name="X")]
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=input_signature)
        return onnx_model

    def _convert_sklearn_to_onnx(self):
        from skl2onnx import convert_sklearn
        initial_types = [("X", FloatTensorType([None, 30]))]
        return convert_sklearn(self.model, initial_types=initial_types)

    def _convert_xgboost_to_onnx(self):
        from onnxmltools import convert_xgboost
        return convert_xgboost(self.model, initial_types=[("X", FloatTensorType([None, 30]))])