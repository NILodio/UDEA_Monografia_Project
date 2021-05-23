from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
import io
from PIL import Image
from models.models import WD_Tranfer_Learning ,SH_Tranfer_Learning


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AvailableModels(Enum):
    __WD_Transfer_Learning = '__WD_Transfer_Learning'
    __SH_Transfer_Learning = '__SH_Transfer_Learning'


models_path = Path('models')
models = {'__WD_Transfer_Learning': WD_Tranfer_Learning(str(models_path / 'wd_tranfer_learning')),
        "__SH_Transfer_Learning" : SH_Tranfer_Learning(str(models_path / 'sh_tranfer_learning'))
    }

@app.post("/{model}")
async def predict(model: AvailableModels, file: bytes = File(...)):
    """
    Serves predictions given an image file.

    Inputs:
        - model: name of the model being used to do the prediction
        - file: image to detect objects
    """
    model = models[model.value]
    img = Image.open(io.BytesIO(file))
    output = model.predict(img)
    return output











