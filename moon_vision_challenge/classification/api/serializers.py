from django.contrib.auth import get_user_model
from rest_framework import serializers

from moon_vision_challenge.classification.models import Inference

User = get_user_model()


# noinspection PyAbstractClass
class NaiveInferenceSerializer(serializers.Serializer):
    model_type = serializers.ChoiceField(choices=Inference.NETWORKS.choices, required=True)
    image = serializers.FileField(required=True)
    result_label = serializers.CharField(allow_blank=True, read_only=True)


class InferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Inference
        fields = ["model_type", "image", "result_label"]
        read_only_fields = ["result_label"]
