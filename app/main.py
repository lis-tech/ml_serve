from fastapi import FastAPI, Body
from pydantic import BaseModel
import pandas as pd

app = FastAPI()


class PredictionInput(BaseModel):
    model_ref: str
    data_to_predict: dict


class PredictionResponse(BaseModel):
    predictions: list


@app.get("/")
async def root():
    return {"message": "ML SERVE API. FLNT, all right reserved"}


@app.post("/predict")
async def predict(prediction_data: PredictionInput = Body(embed=True)) -> dict:
    # Call the predict method of the wrapper
    # res = wrapper.post("/predict", prediction_data)
    return {"msg": "done", "predictions": []}
