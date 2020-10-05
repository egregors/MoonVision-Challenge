from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules
from django.utils.translation import gettext_lazy as _


class ClassificationConfig(AppConfig):
    name = "moon_vision_challenge.classification"
    verbose_name = _("Classification")

    def ready(self):
        try:
            import moon_vision_challenge.classification.signals  # noqa F401
        except ImportError:
            pass

        autodiscover_modules('classificators')
