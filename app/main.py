from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import cache
import dotenv
import uvicorn
import pandas as pd

from models import (
    Settings,
    TextSequenceLabellingResponse,
    TextSequenceLabellingRequest,
    TextClassificationRequest,
    TextClassificationResponse,
    ListModelsResponse,
    GetModelResponse,
    ModelSummary,
    ModelTaskType,
    ModelStatusType,
    ModelBaseLibraryType,
)

dotenv.load_dotenv()

app = FastAPI(
    title="ML Serving API",
    description="Entrypoint for calling Accurait ML models. FLNT, all rights reserved.",
    docs_url="/",
)

CORSMiddleware(
    app,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


@cache
def get_settings():
    return Settings()


@app.post(
    "/predict/text_classification",
    response_model=TextClassificationResponse,
)
def predict_text_classification(
    body: TextClassificationRequest, settings: Settings = Depends(get_settings)
):
    """
    Calls the predict method of the wrapped model.
    """

    return TextClassificationResponse(results=[], confidence_metric="accuracy")


@app.post(
    "/predict/text_sequence_labelling",
    response_model=TextSequenceLabellingResponse,
)
def predict_text_sequence_labelling(
    body: TextSequenceLabellingRequest, settings: Settings = Depends(get_settings)
):
    """
    Calls the predict method of the wrapped model.
    """

    return TextSequenceLabellingResponse(results=[], confidence_metric="accuracy")


@app.get("/registry/models", response_model=ListModelsResponse)
def list_registered_models(settings: Settings = Depends(get_settings)):
    return ListModelsResponse(models=[])


@app.get("/registry/models/{model_id}", response_model=GetModelResponse)
def get_registered_model(model_id: str, settings: Settings = Depends(get_settings)):
    return GetModelResponse(
        model=ModelSummary(
            model_id="model-abc",
            type=ModelTaskType.TEXT_CLASSIFICATION,
            status=ModelStatusType.UNAVAILABLE,
            model_base_library=ModelBaseLibraryType.PYTORCH,
        )
    )


@app.post("/registry/models/{model_id}")
def post_new_model(model_id: str, settings: Settings = Depends(get_settings)):
    ...


@app.delete("/registry/models/{model_id}")
def delete_registered_model(model_id: str, settings: Settings = Depends(get_settings)):
    ...


def main():
    settings = get_settings()
    uvicorn.run(app, port=settings.api_port)


if __name__ == "__main__":
    main()
