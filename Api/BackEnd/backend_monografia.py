# load model api

import os
from requests.api import request
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import json

from classes_monografia import Context, DetectWord, DetectSheet


from PIL import Image


#predict model api
import cv2 
import numpy as np

#fast api
from fastapi import FastAPI,  File, UploadFile

app = FastAPI(title="sheet detection")

Detect_Word = Context(DetectWord())
# Detect_sheet = Context(DetectSheet())

@app.post("/predict_image/{operation}")
async def predict_image(operation, file: UploadFile = File(...),):
    image = np.array(Image.open(file.file))
    
    if operation == "Detect word":
        image_bbox = Detect_Word(image) 
    # elif operation == "Detect sheet":
    #     image_bbox = Detect_sheet(image)
    # elif operation == "detect all":
    #     image = Detect_sheet(image)
    #     image_bbox = Detect_Word(image)

    lists = image_bbox.tolist()
    json_str = json.dumps(lists)

    return {"image":json_str}

 


