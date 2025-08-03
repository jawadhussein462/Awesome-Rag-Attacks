from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
import yaml


class LanguageModelConfig(BaseModel):
    model: str
    model_provider: str
    temperature: float = 0.1
    max_tokens: Optional[int] = 150


class EmbeddingModelConfig(BaseModel):
    model: str
    provider: str


class ChatModelConfig(BaseModel):
    model: str
    model_provider: str
    temperature: float = 0.1
    max_tokens: Optional[int] = 150


class ApiKeyConfiguration(BaseModel):
    api_key: str


class DocumentProcessingParameters(BaseModel):
    max_length: int = 512  # Token limit for document truncation
    top_k: int = 5
    seed: Optional[int] = 42


class DatasetLoaderConfiguration(BaseModel):
    dataset_name: str
    dataset_path: str
    sample_size: int


class RagSystemConfiguration(BaseModel):
    embedding_config: EmbeddingModelConfig
    chat_config: ChatModelConfig
    processing_params: DocumentProcessingParameters = Field(
        default_factory=DocumentProcessingParameters
    )
    save_path: str


class PoisonedRAGAttackConfiguration(BaseModel):
    llm_attack_config: LanguageModelConfig
    num_docs_per_target_query: int
    num_target_queries: int
    num_words_per_doc: int = 10
    seed: int = 42


class ApplicationConfiguration(BaseModel):
    rag_config: RagSystemConfiguration
    dataset_loader_config: DatasetLoaderConfiguration
    poisoned_rag_attack_config: PoisonedRAGAttackConfiguration


CURRENT_DIRECTORY = Path(__file__).resolve().parent
CONFIGURATION_FILE_PATH = CURRENT_DIRECTORY / "config.yaml"


def load_configuration(
    config_path: Path = CONFIGURATION_FILE_PATH
) -> ApplicationConfiguration:
    """Load YAML-based settings and validate them via Pydantic."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        raw_configuration_data = yaml.safe_load(file_handle)
    return ApplicationConfiguration.model_validate(raw_configuration_data)




