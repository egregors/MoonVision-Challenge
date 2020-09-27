from django.contrib.auth import get_user_model
from rest_framework.permissions import AllowAny
from rest_framework.viewsets import ModelViewSet

from .serializers import InferenceSerializer
from ..models import Inference

User = get_user_model()


class NaiveInferenceViewSet(ModelViewSet):
    """ TODO: add doc """
    serializer_class = InferenceSerializer
    queryset = Inference.objects.all()
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        # TODO: straight call of keras application
        return super().create(request, *args, **kwargs)
