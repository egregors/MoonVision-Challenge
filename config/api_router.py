from django.conf import settings
from rest_framework.routers import DefaultRouter, SimpleRouter

from moon_vision_challenge.classification.api.views import NaiveInferenceViewSet

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("inferences", NaiveInferenceViewSet)

app_name = "api"
urlpatterns = router.urls
