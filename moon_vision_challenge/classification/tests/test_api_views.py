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
