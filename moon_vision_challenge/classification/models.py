from django.db import models
from django.utils.translation import gettext_lazy as _


class Inference(models.Model):
    """ Persistent state for async Inferences calculation """

    class STATUS(models.TextChoices):
        """ Status of Inference """
        PENDING = "pending", "waiting for processing"
        PROCESSING = "processing", "processing in progress"
        DONE = "done", "processing is done"

    # TODO:
    #  - [ ] define a path to store?
    #  - [ ] delete images after processing
    image = models.ImageField(_("image"), help_text=_("image to classification"))

    status = models.CharField(_("status"), max_length=16, choices=STATUS.choices, default=STATUS.PENDING)
    label = models.CharField(_("label"), max_length=255,
                             help_text=_("label the image was classified with"))

    class Meta:
        verbose_name = _("inference")
        verbose_name_plural = _("inferences")
