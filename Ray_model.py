import sys
# setting path
#sys.path.append('C:\\Users\\ali.khankan\\Downloads\\notebooks\\Detectron')
from deep_sort import DeepSort
import os, json, cv2, random
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File
from typing import List, Iterable
import time
import numpy as np
from starlette.requests import Request
# import torch
# from torchvision import transforms
# from torchvision.models import resnet18
from PIL import Image

from io import BytesIO

app = FastAPI()

#ray.shutdown()
ray.init(address="auto", namespace="serve")
serve.start(detached=True)

@serve.deployment()
@serve.ingress(app)
class PPEServe:
  def __init__(self):
    print ('Initializing...')
    cfg = get_cfg()
    cfg.merge_from_file("output/config.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.MODEL.DEVICE = 'cuda'
    self.predictor = DefaultPredictor(cfg)
    blank_image = np.zeros((100,100,3), np.uint8)
    self.predictor(blank_image)
    print ('Loading model done')

  @serve.batch()
  async def classify(self, images :List[File]):
    results = []
    for image in images:
      outputs = self.predictor(image)
      outputs = outputs["instances"].to("cpu")
      cls_ids = outputs.pred_classes.tolist()
      cls_conf = outputs.scores.tolist()
      results.append([cls_ids, cls_conf])
    return results
    

  @app.get("/")
  def get(self):
    return "Welcome to the PyTorch model server, this is just a test"
  
  @app.post("/classify_ppe")
  async def classify_image(self, file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #cv2.imwrite('testing.jpg',img)
    return await self.classify(img)

PPEServe.deploy()
