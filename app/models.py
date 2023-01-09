from pydantic import BaseModel, BaseSettings
from pydantic.generics import GenericModel
from typing import Generic, TypeVar, List
from enum import Enum

_ResultT = TypeVar("_ResultT")


class Settings(BaseSettings):
    api_port: int


class TextClassificationRequest(BaseModel):
    model_id: str
    input: List[str]


class TextSequenceLabellingRequest(BaseModel):
    model_id: str
    input: List[str]


class PredictionResponse(GenericModel, Generic[_ResultT]):
    results: List[_ResultT]
    confidence_metric: str


class TextSequenceLabel(BaseModel):
    start: int
    end: int
    text: str
    label: str
    confidence_score: float


TextSequenceLabellingResult = List[TextSequenceLabel]
TextSequenceLabellingResponse = PredictionResponse[TextSequenceLabellingResult]


class TextClassificationLabel(BaseModel):
    label: str
    confidence_score: float


TextClassificationResult = List[TextSequenceLabel]
TextClassificationResponse = PredictionResponse[TextClassificationResult]


class ModelTaskType(Enum):
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_SEQUENCE_LABELLING = "text_sequence_labelling"


class ModelStatusType(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ModelBaseLibraryType(Enum):
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    TENSORFLOW = "tensorflow"


class ModelSummary(BaseModel):
    model_id: str
    type: ModelTaskType
    status: ModelStatusType
    model_base_library: ModelBaseLibraryType


class ListModelsResponse(BaseModel):
    models: List[ModelSummary]


class GetModelResponse(BaseModel):
    model: ModelSummary
