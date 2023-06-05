import model as nn
import datasets as ds
import numpy as np
from fastapi import FastAPI, HTTPException, status, UploadFile
from pydantic import BaseModel
import json
import asyncio

app = FastAPI()

datasets = {"iris": ds.get_iris_dataset,
            "wine": ds.get_wine_dataset,
            "random-sample": ds.generate_random_sample}
models_cache = {}
jobs = {}
# POST /datasets/upload application/octet-stream -> "dataset ID"
# GET /datasets -> ["iris", "random-sample"]
# GET /datasets/{name}
# POST /models -> "model ID"
# GET /models/{id} -> {layers, neurons, state:[ready, training, trained]}
# POST /models/{id}/fit {dataset: <name>} -> 202 Accepted, 404
# POST /models/{id}/predict -> 200, 404


class DatasetName(BaseModel):
    name: str


async def train_model(model: nn.NeuralNetwork, ds_name: DatasetName):
    dataset = datasets[ds_name]()
    x = dataset.x
    y = dataset.y
    num_classes = len(np.unique(y))
    x_train, x_test, y_train, y_test = model.split_data(x, y)
    x_train, _, y_train, _ = model.prepare_data(x_train, x_test, y_train, y_test, num_classes)
    model.fit_model(x_train, y_train)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/datasets")
async def get_datasets():
    return [*datasets]


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile):
    # TODO: persist to volume and add extraction function to datasets dict
    try:
        print("test")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to persist dataset")
    return {}


@app.post("/models", status_code=status.HTTP_201_CREATED)
async def create_model():
    m = nn.NeuralNetwork(2)
    models_cache["alabala"] = {1, 2, "initial", m}
    return {"model_id": "alabala"}


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    if model_id not in models_cache:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')
    m = models_cache[model_id]
    # TODO: serialize model as json
    return m


@app.post("/models/{model_id}/fit", status_code=status.HTTP_202_ACCEPTED)
async def fit_model(model_id: str, dataset: DatasetName):
    if model_id not in models_cache:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')
    m = models_cache[model_id]
    task = asyncio.create_task(train_model(m, dataset))
    return {}
    # TODO: generate job ID
    # json.dumps(nparray.tolist())
