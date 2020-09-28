from django.contrib.auth import get_user_model
from rest_framework import mixins, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from .serializers import NaiveInferenceSerializer

User = get_user_model()


# TODO: extract it to module with keras
def get_label_for_img(model_type, image):
    raise NotImplemented()


class NaiveInferenceViewSet(mixins.CreateModelMixin,
                            GenericViewSet):
    """ TODO: add doc """
    serializer_class = NaiveInferenceSerializer
    # TODO: change perms for prod
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # nothing to save
        # self.perform_create(serializer)

        data = serializer.validated_data
        mt = data["model_type"]
        img = data["image"]

        # TODO: make classification here
        result_label = get_label_for_img(model_type=mt, image=img)

        headers = self.get_success_headers(serializer.data)
        return Response(
            {
                'result_label': result_label
            },
            status=status.HTTP_200_OK,
            headers=headers
        )
