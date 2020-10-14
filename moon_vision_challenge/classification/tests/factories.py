import factory


class InferenceFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'classification.Inference'
