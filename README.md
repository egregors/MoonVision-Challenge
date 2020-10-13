# Django REST endpoint for image classification

### Models-serving
For models serving is used [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) 
with two pre-trained models: [resnet and inception](https://moon-vision-demo.s3.eu-central-1.amazonaws.com/model-data.tar.gz) 
according `models.config`:

```
model_config_list {
  config {
    name: 'resnet'
    base_path: '/models/resnet/'
    model_platform: 'tensorflow'
  }
  config {
    name: 'inception'
    base_path: '/models/inception/'
    model_platform: 'tensorflow'
  }
}
```

### Core

The main idea for my architect decision is to give a consumer a way to simple and flexible work with different model types.
For this purpose, I have implemented a few abstractions for describing `Classificator` entity, which provides a common interface 
for all `Classificators` and `Dispatcher` to dynamic dispatching requests inside API view or async tasks.

The whole system represents in [local.yml](https://github.com/egregors/MoonVision-Challenge/blob/master/moon_vision_challenge/local.yml)
and contain 7 services: 

* `tensorflow-serving` – container with official `tensorflow/serving`
* `django` – the main app 
* `postgres` – persistent store
* `redis` – kv-store for queue broker
* `celeryworker` – queue worker
* `celerybeat` – worker for cron-like tasks
* `flower` – tool for queue monitoring
* **(production)** `traefik` – reverse proxy with SSL

#### Classificator

The abstract `Classificator` implementation could be found in [abstract_classificator.py](https://github.com/egregors/MoonVision-Challenge/blob/master/moon_vision_challenge/classification/abstract_classificator.py)
A consumer should set `model_type`, `model_url` (internal tensorflow-serving URL) and implement all required methods: 

* `preprocess_input(self, image: Image) -> Any` – prepare an image to classification
* `process_prediction(self, input_vector: Any) -> Dict` – perform a classification
* `decode_predictions(self, pred: Dict) -> str` – decode classification result to label

After that any inheritor could be registered in the dispatcher. For example:
```python
from typing import Any, Dict

from PIL import Image

from .abstract_classificator import Classificator
from .dispatchers import dispatcher as classificators


class ResNetClassificator(Classificator):
    model_type = 'resnet'
    model_url = 'http://tensorflow-serving:8501/v1/models/resnet:predict'

    def preprocess_input(self, image: Image) -> Any:
        # ...

    def process_prediction(self, input_vector: Any) -> Dict:
        # ...

    def decode_predictions(self, pred: Dict) -> str:
        # ...

# registering new classificator
classificators.register(ResNetClassificator)
```

That's all! All registered classificators will be able for dispatching after Django bootstrap by auto-discovery mechanism. 

Check out the full examples in [classificators.py](https://github.com/egregors/MoonVision-Challenge/blob/master/moon_vision_challenge/classification/classificators.py)

#### Dispatcher

A `Dispatcher` is an abstraction that auto-discover (`classificators.py` files in any app) and contain all registered `Classificators`.
This way provides dynamic defining full lifecycle of prediction according to the chosen model type.

An implementation of `Dispatcher` could be found in [dispatchers.py](https://github.com/egregors/MoonVision-Challenge/blob/master/moon_vision_challenge/classification/dispatchers.py)

Like a `Classificator`, `Dispatcher` expose base abstract class as well. It gives a consumer way to define their own dispatcher.

### REST API

There are an implementation of two different ways to solve this problem. 

#### v0 – Naive synchronous entrypoint

`Allow: POST, OPTIONS`
`Content-Type: application/json`

The naive entry point is absolutely stateless, synchronous and straightforward. 
This entry point just expects valid `model_type` and `image` in the request payload and response with `[200]` prediction label (according dispatcher) or `[4/5xx]` error with details.

The performance of this kind of view would dramatically depend on: hardware, way to WSGI Django serving, performance of `tensorflow-serving`, requests capacity, etc.
This way is pretty hard to scaling as well, even it's still stateless view.

Check this view out in [views.py](https://github.com/egregors/MoonVision-Challenge/blob/master/moon_vision_challenge/classification/api/views.py)

* `OPTIONS /api/v0/inferences/` – allows consumers to get all information about entry point signature and required fields.

* `POST /api/v0/inferences/` – perform a prediction label for an image by synchronous view according to chosen model type.

Full entry point signature:

```json
{
    "model_type": {
        "type": "choice",
        "required": true,
        "read_only": false,
        "label": "Model type",
        "choices": [
            {
                "value": "resnet",
                "display_name": "resnet"
            },
            {
                "value": "inception",
                "display_name": "inception"
            }
        ]
    },
    "image": {
        "type": "file upload",
        "required": true,
        "read_only": false,
        "label": "Image"
    },
    "label": {
        "type": "string",
        "required": false,
        "read_only": true,
        "label": "Label"
    }
}
```

Payload example:
```json5
{
    "model_type": "resnet", // a string signifying the type of pretrained model
    "image": { Image }      // an image binary 
}
```

Response example:
```json5
{
    "label": "albatross"
}
```

#### v1 – Asynchronous entrypoint with queue

`Allow: POST, OPTIONS`
`Content-Type: application/json`

The async entry point is creating `Celery` task for each request and put it in the queue. Then, `celeryworker's` receive tasks and perform prediction
as usual (by a dispatcher). Each `Inference.status` could be:

```python
class STATUS(models.TextChoices):
    """ Status of Inference """
    PENDING = "pending", "waiting for processing"
    PROCESSING = "processing", "processing in progress"
    FAIL = "fail", "can't be processed"
    DONE = "done", "processing is done"
```

Details of `Inference` model could be found in [models.py](https://github.com/egregors/MoonVision-Challenge/blob/master/moon_vision_challenge/classification/models.py)

This entry point expect the same input payload format as `api/v0`, however much more suitable for scaling, providing data security, monitoring.


* `OPTIONS /api/v1/inferences/` – allows consumers to get all information about entry point signature and required fields.

* `POST /api/v1/inferences/` – perform a creation of async task for prediction for an image according to chosen model type.

The entry point signature and input payload format are absolutely the same as for naive way. 
However, response will contain async inference details and status code `201`: 

```json5
{
    "id": 1,                                                                        // id of inference (need to get result)
    "model_type": "resnet",                                                         
    "image": "http://0.0.0.0:8000/media/slider_puffin_before_mobile_gmAnaQd.jpg",
    "label": "",                                                                    // empty label until "status" is not "done"
    "status": "pending"                                                             // current status of task
}
```

* `GET /api/v1/inferences/{inference_id:int}/` – entry point to get results of async prediction task.

For example: 

```json5
{
    "id": 1,
    "model_type": "resnet",
    "image": "http://0.0.0.0:8000/media/slider_puffin_before_mobile_gmAnaQd.jpg",
    "label": "albatross",
    "status": "done"
}
```

### Future improvements

I believe the second solution (`api/v1/` async view) is already have good potential for scaling. 
Though I should add caching for view, pay more attention to throttling configuration, and add more tooling for consumers.

## Build

* Download pre-trained models and build containers: `make build`

* [Local environment] Create all services, up them like a demon and show logs: `make up`

* [Local environment] Stop all services: `make down`

To get quick help use `make`:

```shell script
➜  MoonVision-Challenge git:(master) ✗ make
Usage: make [task]

task                 help
------               ----
                     
get_tf_models        Download and unzip tf-models
build                Get TF models and build services
up                   Run server and shows logs
down                 Stop services
lint                 Lint and type check
test                 Run tests
                     
help                 Show help message
```

### Lint | Tests

Run linter and mypy: `make lint`

Run tests: `make test`
