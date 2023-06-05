import model as nn
import datasets as ds
import numpy as np
from fastapi import FastAPI, HTTPException, status, UploadFile, Response
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
    return True


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
        async with aiofiles.open(f'/app/data/datasets/{file.filename}', mode='wb', buffering=4096) as f:
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
    models_cache[model_id] = {"layers": m.num_layers,
                              "neurons": m.num_neurons_per_layer,
                              "state": "initial",
                              "obj": m}
    return {"model_id": model_id}


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    if model_id not in models_cache:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')

    m = models_cache[model_id]
    return {"layers": m.layers, "neurons": m.neurons, "state": m.state}


@app.post("/models/{model_id}/fit", status_code=status.HTTP_202_ACCEPTED)
async def fit_model(model_id: str, ds_name: DatasetName):
    if model_id not in models_cache:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')

    if ds_name not in datasets:
        raise HTTPException(status_code=400, detail=f'Invalid dataset: {ds_name}')

    if model_id in jobs:
        raise HTTPException(status_code=400, detail=f'Model {ds_name} training already started')

    m = models_cache[model_id]
    task = asyncio.create_task(train_model(m, ds_name))
    jobs[model_id] = task
    return {"status": "started training"}


@app.post("/models/{model_id}/predict")
async def model_predict(model_id: str, ds_name: DatasetName, response: Response):
    if model_id not in models_cache:
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')

    if ds_name not in datasets:
        raise HTTPException(status_code=400, detail=f'Invalid dataset: {ds_name}')

    if model_id not in jobs:
        raise HTTPException(status_code=400, detail=f'Model {ds_name} not trained')

    job = jobs[model_id]
    if not job.done():
        response.status_code = 204
        return {"status": "training"}

    try:
        _ = job.result()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Model {model_id} training error: {e}')

    m = models_cache[model_id]
    model = m.obj
    ds = datasets[ds_name]()
    x = ds.x
    y = ds.y
    num_classes = len(np.unique(y))
    x_train, x_test, y_train, y_test = model.split_data(x, y)
    _, x_test, _, _ = model.prepare_data(x_train, x_test, y_train, y_test, num_classes)

    result = model.predict(x_test)
    return json.dumps(result.tolist())
