import model as nn
import datasets as ds
import numpy as np
from fastapi import FastAPI, HTTPException, status, UploadFile
from pydantic import BaseModel
import json
import asyncio
import uuid
import aiofiles

app = FastAPI()

datasets = {"iris": ds.get_iris_dataset,
            "wine": ds.get_wine_dataset,
            "random-sample": ds.generate_random_sample}
models_cache = {}
jobs = {}
# POST /datasets/upload -> "dataset ID"
# GET /datasets -> ["iris", "random-sample"]
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
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a csv")

    try:
        async with aiofiles.open(f'datasets/{file.filename}', mode='wb', buffering=4096) as f:
            while buff := await file.read(4096):
                await f.write(str(buff))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'failed to persist dataset: {e}')
    finally:
        await file.close()

    return {"status": "uploaded"}


@app.post("/models", status_code=status.HTTP_201_CREATED)
async def create_model(num_classes: int):
    m = nn.NeuralNetwork(num_classes)
    model_id = str(uuid.uuid4())
    models_cache[model_id] = {m.num_layers, m.num_neurons_per_layer, "initial", m}
    return {"model_id": model_id}


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
    return {"status": "started training"}
    # TODO: generate job ID
    # json.dumps(nparray.tolist())
