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
    """ Default Dispatcher for Classificators """
    _classificators: Dict[str, Type[Classificator]]

    def __str__(self) -> str:
        return ', '.join(self._classificators)

    def __getitem__(self, model_type: str) -> Type[Classificator]:
        return self._classificators[model_type]

    def _classificator_is_valid(self, classificator: Type[Classificator]):
        # TODO: add check of imp all abs methods
        return True

    def register(self, classificator: Type[Classificator]):
        if self._classificator_is_valid(classificator):
            logger.info('registering action `%s`', classificator.model_type)
            self._classificators[classificator.model_type] = classificator
