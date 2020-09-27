from django.db import models
from django.utils.translation import gettext_lazy as _


class Inference(models.Model):
    """ TODO: add docstring """

    class NETWORKS(models.TextChoices):
        """ Pre-trained keras models """
        DenseNet121 = "DenseNet121", "DenseNet121"
        DenseNet169 = "DenseNet169", "DenseNet169"
        DenseNet201 = "DenseNet201", "DenseNet201"
        InceptionResNetV2 = "InceptionResNetV2", "InceptionResNetV2"
        InceptionV3 = "InceptionV3", "InceptionV3"
        MobileNet = "MobileNet", "MobileNet"
        MobileNetV2 = "MobileNetV2", "MobileNetV2"
        NASNetLarge = "NASNetLarge", "NASNetLarge"
        NASNetMobile = "NASNetMobile", "NASNetMobile"
        ResNet101 = "ResNet101", "ResNet101"
        ResNet101V2 = "ResNet101V2", "ResNet101V2"
        ResNet152 = "ResNet152", "ResNet152"
        ResNet152V2 = "ResNet152V2", "ResNet152V2"
        ResNet50 = "ResNet50", "ResNet50"
        ResNet50V2 = "ResNet50V2", "ResNet50V2"
        VGG16 = "VGG16", "VGG16"
        VGG19 = "VGG19", "VGG19"
        Xception = "Xception", "Xception"

    model_type = models.CharField(_("the type of pretrained model"), max_length=64, choices=NETWORKS.choices)
    # TODO:
    #  - [ ] define a path to store?
    #  - [ ] delete images after processing
    image = models.ImageField(_("image"), help_text=_("image to classification"))
    result_label = models.CharField(_("label"), max_length=255,
                                    help_text=_("label the image was classified with"))

    class Meta:
        verbose_name = _("inference")
        verbose_name_plural = _("inferences")
