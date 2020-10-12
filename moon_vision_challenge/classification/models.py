from django.db import models
from django.utils.translation import gettext_lazy as _

from moon_vision_challenge.classification.dispatchers import dispatcher as classificators


class Inference(models.Model):
    """ Persistent state for async Inferences calculation """

    class STATUS(models.TextChoices):
        """ Status of Inference """
        PENDING = "pending", "waiting for processing"
        PROCESSING = "processing", "processing in progress"
        FAIL = "fail", "can't be processed"
        DONE = "done", "processing is done"

    image = models.ImageField(_("image"), help_text=_("image to classification"))

    status = models.CharField(_("status"), max_length=16, choices=STATUS.choices, default=STATUS.PENDING)
    label = models.CharField(_("label"), max_length=255,
                             help_text=_("label the image was classified with"))

    task_id = models.CharField(_("task id"), max_length=64, null=True, default=None, unique=True)
    model_type = models.CharField(_("model type"), max_length=64,
                                  choices=[(t, t) for t in classificators.get_model_types()])

    class Meta:
        verbose_name = _("inference")
        verbose_name_plural = _("inferences")
