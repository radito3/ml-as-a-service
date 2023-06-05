import model as nn
from fastapi import FastAPI, HTTPException

app = FastAPI()

models_cache = {}

# POST /datasets/upload application/octet-stream -> "dataset ID"
# GET /datasets -> ["iris", "random-sample"]
# GET /datasets/{name}
# POST /models -> "model ID"
# GET /models/{id} -> {layers, neurons, state:[ready, training, trained]}
# POST /models/{id}/fit {dataset: <name>} -> 202 Accepted, 404
# POST /models/{id}/predict -> 200, 404


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/datasets")
async def get_datasets():
    return ["iris", "random-sample"]


@app.post("/models")
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
