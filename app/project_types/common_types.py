from pydantic import BaseSettings
from enum import Enum


class Settings(BaseSettings):
    api_port: int


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
