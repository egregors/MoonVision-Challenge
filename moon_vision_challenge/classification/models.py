from django.db import models
from django.utils.translation import gettext_lazy as _

from moon_vision_challenge.classification.dispatchers import dispatcher as classificators


class Inference(models.Model):
    """ TODO: add docstring """

    class STATUS(models.TextChoices):
        """ Status of Inference """
        PENDING = "pending", "waits for processing"
        PROCESSING = "processing", "processing in progress"
        DONE = "done", "processing is done"

    model_type = models.CharField(_("the type of pretrained model"), max_length=64,
                                  choices=classificators.get_model_types())
    # TODO:
    #  - [ ] define a path to store?
    #  - [ ] delete images after processing
    image = models.ImageField(_("image"), help_text=_("image to classification"))

    status = models.CharField(_("status"), max_length=16, choices=STATUS.choices, default=STATUS.PENDING)
    result_label = models.CharField(_("label"), max_length=255,
                                    help_text=_("label the image was classified with"))

    class Meta:
        verbose_name = _("inference")
        verbose_name_plural = _("inferences")
