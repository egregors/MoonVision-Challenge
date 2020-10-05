import logging
from abc import ABC, abstractmethod
from typing import Any

from django.core.files.uploadedfile import InMemoryUploadedFile

logger = logging.getLogger(__name__)


class Classificator(ABC):
    """ The base classificator meta class. All implementations should be inheritance from this one.
    Required implementation `preprocess_input` and `decode_predictions` methods.
    """

    model_type: str

    @abstractmethod
    def preprocess_input(self, image: InMemoryUploadedFile) -> Any:
        """
        Convert Image to proper type according ML-model signature.

        :param image: InMemoryUploadedFile object from request
        :return: Object which expected by ML-model
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_predictions(self) -> Any:
        """
        Decode predictions from a model, and convert it to useful type.

        :rtype: Any (consumer choice)
        :return: Representation of ML-model predictions
        """
        raise NotImplementedError()
