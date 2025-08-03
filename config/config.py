from typing import Dict
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Literal
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


class CorruptRAGAttackConfiguration(BaseModel):
    """Configuration for CorruptRAG attack - single-document poisoning."""
    llm_attack_config: LanguageModelConfig
    num_target_queries: int
    num_words_per_doc: int = 250  # Larger documents for better stealth
    seed: int = 42
    attack_method: Literal["AS", "AK"] = "AS"  # AS or AK method
    max_refinement_attempts: int = 3  # Max refinement attempts for AK
    ak_word_limit: int = 50  # Word limit for AK refined corpus (V parameter)


class AttackConfiguration(BaseModel):
    poisoned_rag_attack_config: PoisonedRAGAttackConfiguration
    corrupt_rag_attack_config: CorruptRAGAttackConfiguration
    

class ApplicationConfiguration(BaseModel):
    rag_config: RagSystemConfiguration
    dataset_loader_config: DatasetLoaderConfiguration
    attack_config: AttackConfiguration


CURRENT_DIRECTORY = Path(__file__).resolve().parent
CONFIGURATION_FILE_PATH = CURRENT_DIRECTORY / "config.yaml"


def load_configuration(
    config_path: Path = CONFIGURATION_FILE_PATH
) -> ApplicationConfiguration:
    """Load YAML-based settings and validate them via Pydantic."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        raw_configuration_data = yaml.safe_load(file_handle)
    return ApplicationConfiguration.model_validate(raw_configuration_data)




