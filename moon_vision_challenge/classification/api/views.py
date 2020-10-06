import logging

from PIL import Image
from rest_framework import mixins, status
from rest_framework.exceptions import APIException
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from .serializers import NaiveInferenceSerializer, InferenceSerializer
from ..dispatchers import dispatcher as classificators
from ..tasks import run_inference

logger = logging.getLogger(__name__)


# TODO: add throttling
class NaiveInferenceViewSet(mixins.CreateModelMixin, GenericViewSet):
    """ Naive way to make quick prediction. This is a synchronous view. """

    serializer_class = NaiveInferenceSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        model_type = serializer.validated_data['model_type']
        image = Image.open(serializer.validated_data['image'])

        try:
            label = classificators[model_type](image=image).predict()
        except Exception as err:
            logger.error("can't make prediction: %s", err)
            raise err
            raise APIException("can't make prediction: %s" % err) from err

        headers = self.get_success_headers(serializer.data)
        return Response(
            {'label': label}, status=status.HTTP_200_OK, headers=headers
        )
