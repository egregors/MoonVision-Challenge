from time import sleep

from PIL import Image
from celery import Task
from django.contrib.auth import get_user_model

from config import celery_app
from moon_vision_challenge.classification.dispatchers import dispatcher as classificators
from moon_vision_challenge.classification.models import Inference

User = get_user_model()


@celery_app.task()
def get_users_count():
    """A pointless Celery task to demonstrate usage."""
    return User.objects.count()


@celery_app.task(bind=True)
def run_inference(self: Task, inference_id: int):
    inference = Inference.objects.get(pk=inference_id)
    inference.task_id = self.request.id
    inference.status = Inference.STATUS.PROCESSING
    inference.save()

    self.update_state(state=Inference.STATUS.PROCESSING)
    pil_image = Image.open(inference.image)
    inference.label = classificators[inference.model_type](pil_image).predict()
    inference.status = Inference.STATUS.DONE
    inference.save()

    self.update_state(state=Inference.STATUS.DONE)
