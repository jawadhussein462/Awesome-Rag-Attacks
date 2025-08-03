"""
Abstract Base Class for RAG Attacks

This module defines the abstract base class that all RAG attack implementations
should inherit from. It provides a common interface for attack classes while
allowing for different implementations.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain.chat_models import init_chat_model


class BaseRAGAttack(ABC):
    """
    Abstract base class for RAG attack implementations.
    
    This class defines the common interface that all attack implementations
    must follow, ensuring consistency across different attack types.
    
    Attributes:
        attack_configuration: Configuration for the attack parameters
        adversarial_llm_config: Configuration for the language model used
        adversarial_language_model: The LLM instance for generating content
    """

    def __init__(self, attack_config):
        """
        Initialize the attack system with configuration.
        
        Args:
            attack_config: Configuration containing attack parameters,
                including LLM settings.
        """
        self.attack_configuration = attack_config
        self.seed = attack_config.seed
        self.num_target_queries = attack_config.num_target_queries
        self.adversarial_llm_config = (
            self.attack_configuration.llm_attack_config
        )
        self.adversarial_language_model = init_chat_model(
            **self.adversarial_llm_config.model_dump()
        )

    @abstractmethod
    def create_misleading_answer(self, target_query: str) -> str:
        """
        Generate an incorrect answer for a target query.
        
        Args:
            target_query: The query for which to generate an incorrect answer
            
        Returns:
            A string containing an incorrect but plausible answer
        """
        pass

    @abstractmethod
    def generate_malicious_corpus_for_target_queries(
        self,
        target_query_list: List[str],
    ) -> List[str]:
        """
        Generate malicious documents for multiple target queries.
        
        Args:
            target_query_list: List of queries to generate malicious 
                documents for
            
        Returns:
            List of malicious document strings for all target queries
        """
        pass 