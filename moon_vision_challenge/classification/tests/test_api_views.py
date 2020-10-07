from rest_framework import status
from rest_framework.test import APITestCase


class NaiveInferenceTestCase(APITestCase):
    def test_allowed_methods(self):
        r = self.client.get(path='/api/v0/inferences/')
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.put(path='/api/v0/inferences/')
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.patch(path='/api/v0/inferences/')
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.delete(path='/api/v0/inferences/')
        self.assertEqual(r.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        r = self.client.options(path='/api/v0/inferences/')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
