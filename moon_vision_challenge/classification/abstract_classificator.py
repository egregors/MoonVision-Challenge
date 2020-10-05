import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Classificator(ABC):
    """ The base classificator meta class. All implementations should be inheritance from this one.

    TODO: add docs for abs methods
    """

    model_type: str

    @abstractmethod
    def preprocess_input(self):
        raise NotImplementedError()

    @abstractmethod
    def decode_predictions(self):
        raise NotImplementedError()
