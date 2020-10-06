from django.conf import settings
from rest_framework.routers import DefaultRouter, SimpleRouter

from moon_vision_challenge.classification.api.views import NaiveInferenceViewSet, InferenceViewSet

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("v0/inferences", NaiveInferenceViewSet, basename="naive-inferences")
router.register("v1/inferences", InferenceViewSet, basename="inferences")

app_name = "api"
urlpatterns = router.urls
