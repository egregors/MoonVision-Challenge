import base64
import json
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from keras.utils import data_utils
from rest_framework import status

from .abstract_classificator import Classificator
from .dispatchers import dispatcher as classificators
from .exceptions import TensorFlowServingException

IMAGENET_CLASS_INDEX = None
IMAGENET_CLASS_INDEX_PATH = ('https://storage.googleapis.com/download.tensorflow.org/'
                             'data/imagenet_class_index.json')


class ResNetClassificator(Classificator):
    model_type = 'resnet'
    model_url = 'http://tensorflow-serving:8501/v1/models/resnet:predict'

    _resize_size = (224, 224)

    def preprocess_input(self, image: InMemoryUploadedFile) -> Any:
        pil_image: Image.Image = Image.open(image).resize(size=self._resize_size)

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return b64_image

    def process_prediction(self, input_vector: Any) -> Any:
        payload = {
            "instances": [{'b64': input_vector}]
        }

        return requests.post(
            url=self.model_url,
            json=payload
        )

    def decode_predictions(self, pred: Any) -> Any:
        global IMAGENET_CLASS_INDEX

        if IMAGENET_CLASS_INDEX is None:
            fpath = data_utils.get_file(
                'imagenet_class_index.json',
                IMAGENET_CLASS_INDEX_PATH,
                cache_subdir='models',
                file_hash='c2c37ea517e94d9795004a39431a14cb')
            with open(fpath) as f:
                IMAGENET_CLASS_INDEX = json.load(f)

        class_ = str(json.loads(pred.content.decode('utf-8'))['predictions'][0]['classes'])
        return IMAGENET_CLASS_INDEX[class_][1]


class InceptionClassificator(Classificator):
    model_type = 'inception'
    model_url = 'http://tensorflow-serving:8501/v1/models/inception:predict'

    _resize_size = (224, 224)

    def preprocess_input(self, image: InMemoryUploadedFile) -> Any:
        pil_image: Image.Image = Image.open(image).resize(size=self._resize_size)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return b64_image

    def process_prediction(self, input_vector: Any) -> Any:
        payload = {
            "signature_name": "predict_images",
            "instances": [{'b64': input_vector}]
        }

        r = requests.post(url=self.model_url, json=payload)

        if r.status_code != status.HTTP_200_OK:
            raise TensorFlowServingException("Wrong Response from TF-serving: %s", r.content.decode('utf-8'))

        return json.loads(r.content.decode('utf-8'))

    def decode_predictions(self, pred: Any) -> Any:
        class_ = str(json.loads(pred.content.decode('utf-8'))['predictions'][0]['classes'])
        return class_[0]


classificators.register(ResNetClassificator)
classificators.register(InceptionClassificator)
