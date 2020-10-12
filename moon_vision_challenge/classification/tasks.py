import logging

from PIL import Image
from celery import Task

from config import celery_app
from moon_vision_challenge.classification.dispatchers import dispatcher as classificators
from moon_vision_challenge.classification.models import Inference

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def run_inference(self: Task, inference_id: int):

    inference = Inference.objects.get(pk=inference_id)
    inference.task_id = self.request.id
    inference.status = Inference.STATUS.PROCESSING
    inference.save()

    self.update_state(state=Inference.STATUS.PROCESSING)

    try:
        pil_image = Image.open(inference.image)
        inference.label = classificators[inference.model_type](pil_image).predict()
    except Exception as err:
        logger.error("can't get prediction: %s", err)
        inference.status = Inference.STATUS.FAIL
        inference.save()
        return

    inference.status = Inference.STATUS.DONE
    inference.save()

    self.update_state(state=Inference.STATUS.DONE)
