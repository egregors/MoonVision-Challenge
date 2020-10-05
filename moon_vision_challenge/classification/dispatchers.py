import logging
from abc import ABC, abstractmethod
from typing import Dict, Type

from moon_vision_challenge.classification.abstract_classificator import Classificator

logger = logging.getLogger(__name__)


class BaseClassificatorDispatcher(ABC):
    """ Meta class for all Dispatchers """

    @abstractmethod
    def register(self, classificator: Classificator):
        raise NotImplementedError()


class DefaultClassificatorDispatcher(BaseClassificatorDispatcher):
    """ Default Dispatcher for Classificators. """

    _classificators: Dict[str, Type[Classificator]] = {}

    def __str__(self) -> str:
        return ', '.join(self._classificators)

    def __getitem__(self, model_type: str) -> Type[Classificator]:
        return self._classificators[model_type]

    def has_classificator(self, model_type: str) -> bool:
        """ Check if Dispatcher has handler for particular model type classificator. """
        return model_type in self._classificators

    def _classificator_is_valid(self, classificator: Type[Classificator]) -> bool:
        if not classificator.model_type:
            raise ValueError('`model_type` attribute is required for all Classificators')

        if not classificator.model_url:
            raise ValueError('`model_url` attribute is required for all Classificators')

        if self.has_classificator(classificator.model_type):
            raise ValueError('`{0}` model type already registered'.format(classificator.model_type))

        if getattr(classificator.preprocess_input, '__isabstractmethod__', False):
            raise ValueError('`preprocess_input` method is required')

        if getattr(classificator.decode_predictions, '__isabstractmethod__', False):
            raise ValueError('`decode_predictions` method is required')

        return True

    def register(self, classificator: Type[Classificator]):
        if self._classificator_is_valid(classificator):
            logger.info('registering classificator `%s`', classificator.model_type)
            self._classificators[classificator.model_type] = classificator


dispatcher = DefaultClassificatorDispatcher()

__all__ = ['dispatcher']
