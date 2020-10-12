import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from PIL.Image import Image

logger = logging.getLogger(__name__)


class Classificator(ABC):
    """
    The base classificator meta class. All implementations should be inheritance from this one.
    Required implementation `preprocess_input`, 'process_prediction' and `decode_predictions` methods.

    Each Classificator must define
        `model_type` – the type of pretrained model, like resnet, vgg, mobilenet, etc;
        `model_url` – url to access tensorflow-serving API for particular model.
    """

    # TODO: URL could be get by concatenate BASE_MODEL_URL and model_type
    model_type: str
    model_url: str

    def __init__(self, image: Image):
        self.image = image

    @abstractmethod
    def preprocess_input(self, image: Image) -> Any:
        """
        Convert Image to proper type according ML-model signature.

        :param image: PIL Image object
        :return: An object which expected by ML-model
        """
        raise NotImplementedError()

    @abstractmethod
    def process_prediction(self, input_vector: Any) -> Dict:
        """
        Define a payload for request to TF-serving REST API.
        Payload should contain fields according model input signature.

        :param input_vector: An object which expected by ML-model
        :return: Unmarshalled JSON response from TF-serving REST API
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_predictions(self, pred: Dict) -> str:
        """
        Decode predictions from a model, and convert it to string.

        :return: Representation of ML-model predictions
        """
        raise NotImplementedError()

    def predict(self) -> Any:
        return self.decode_predictions(self.process_prediction(self.preprocess_input(self.image)))
