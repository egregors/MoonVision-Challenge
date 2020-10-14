from django.conf import settings
from rest_framework import status
from rest_framework.test import APITestCase

from moon_vision_challenge.classification.tests.factories import InferenceFactory


class NaiveInferenceTestCase(APITestCase):
    inferences_url = '/api/v0/inferences/'

    def test_allowed_methods(self):
        r = self.client.get(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.put(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.patch(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.delete(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.options(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_200_OK)

        r = self.client.post(path=self.inferences_url, data={}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_bad_image_format(self):
        payload = {
            "model_type": "resnet",
            "image": open(settings.APPS_DIR / 'classification/tests/fixtures/bad_boy.jpg', 'rb')
        }
        r = self.client.post(self.inferences_url, data=payload, format='multipart')
        self.assertEqual(r.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(r.data['detail'], "can't make prediction: cannot write mode RGBA as JPEG")

    def test_success_predict(self):
        payload = {
            "model_type": "resnet",
            "image": open(settings.APPS_DIR / 'classification/tests/fixtures/cat.jpg', 'rb')
        }
        r = self.client.post(self.inferences_url, data=payload, format='multipart')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['label'], "tiger_cat")


class AsyncInferenceTestCase(APITestCase):
    inferences_url = '/api/v1/inferences/'

    def test_allowed_methods(self):
        r = self.client.get(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.put(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.patch(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.delete(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.options(path=self.inferences_url)
        self.assertEqual(r.status_code, status.HTTP_200_OK)

        i = InferenceFactory()
        r = self.client.get(path=f"{self.inferences_url}{i.id}/")
        self.assertEqual(r.status_code, status.HTTP_200_OK)
