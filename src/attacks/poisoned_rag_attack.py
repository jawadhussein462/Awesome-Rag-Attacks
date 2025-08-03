"""
PoisonedRAG Attack Implementation

This module implements a simplified version of the PoisonedRAG attack against 
Retrieval-Augmented Generation (RAG) systems. The attack works by generating 
malicious documents that are designed to mislead the RAG system when specific 
target queries are made.

The attack consists of two main components:
1. Generator Attack: Creates documents with incorrect information that would 
   lead to wrong answers for target queries
2. Retrieval Attack: Creates documents optimized for high retrieval relevance 
   while containing misleading content

Based on the research paper: "PoisonedRAG: Knowledge Poisoning Attacks on 
Retrieval-Augmented Generation"
"""

from typing import List

from config.config import PoisonedRAGAttackConfiguration
from langchain.chat_models import init_chat_model
from src.prompts import (
    ADVERSARIAL_INCORRECT_ANSWER_PROMPT,
    ADVERSARIAL_DOCUMENT_GENERATION_PROMPT
)


class PoisonedRAG:
    """
    Implements PoisonedRAG attacks against RAG systems.

    This class generates malicious documents designed to mislead RAG systems
    when specific target queries are made. The attack uses a language model
    to create adversarial content that appears relevant but contains incorrect
    information.

    Attributes:
        attack_configuration: Configuration for the attack parameters
        llm_attack_config: Configuration for the language model used in attacks
        adversarial_language_model: The language model instance used to generate malicious content
    """

    def __init__(
        self,
        poisoned_rag_attack_config: PoisonedRAGAttackConfiguration
    ):
        """
        Initialize the PoisonedRAG attack system.
        
        Args:
            poisoned_rag_attack_config: Configuration containing attack parameters,
                including LLM settings, number of documents per query, etc.
        """
        self.attack_configuration = poisoned_rag_attack_config
        self.adversarial_llm_config = self.attack_configuration.llm_attack_config
        self.adversarial_language_model = init_chat_model(**self.adversarial_llm_config.model_dump())

    def create_misleading_answer(
        self,
        target_query: str,
    ) -> str:
        """
        Generate an incorrect answer for a target query.
        
        This method uses the attack language model to generate a plausible but
        incorrect answer for the given query. This incorrect answer will be
        embedded in malicious documents to mislead the RAG system.
        
        Args:
            target_query: The query for which to generate an incorrect answer
            
        Returns:
            A string containing an incorrect but plausible answer
        """
        adversarial_prompt = ADVERSARIAL_INCORRECT_ANSWER_PROMPT.format(target_query=target_query)
        response = self.adversarial_language_model.invoke(adversarial_prompt)
        return response.content
       
    def create_generator_attack_document(
        self,
        target_query: str,
        misleading_answer: str,
    ) -> str:
        """
        Generate an adversarial document designed to mislead a specific query.
        
        This creates a document containing the target query and incorrect information
        that would lead to the wrong answer when retrieved by the RAG system.
        
        Args:
            target_query: The query to target with the adversarial document
            misleading_answer: The incorrect answer to embed in the document
            
        Returns:
            Adversarial document string designed to mislead the RAG system
        """
        
        # Create adversarial content that appears relevant but contains
        # wrong information
        adversarial_document_prompt = ADVERSARIAL_DOCUMENT_GENERATION_PROMPT.format(
            target_query=target_query,
            misleading_answer=misleading_answer,
            num_words=self.attack_configuration.num_words_per_doc
        )

        response = self.adversarial_language_model.invoke(adversarial_document_prompt)
        return response.content

    def create_retrieval_attack_document(
        self,
        target_query: str,
    ) -> str:
        """
        Create a retrieval attack document.
        
        This method creates a document designed to be highly relevant for retrieval
        while containing misleading information.
        
        Args:
            target_query: The query to optimize retrieval for
            
        Returns:
            Document string optimized for retrieval but containing misleading content
        """

        return target_query

    def generate_malicious_document_corpus(
        self,
        target_query: str,
    ) -> List[str]:
        """
        Generate a malicious corpus for a target query.
        
        Args:
            target_query: The query to generate malicious documents for
            
        Returns:
            List of malicious document strings designed to mislead the RAG system
        """
        adversarial_document_scenarios = []
        misleading_answer = self.create_misleading_answer(target_query)

        for _ in range(self.attack_configuration.num_docs_per_target_query):

            generator_attack_document = self.create_generator_attack_document(
                target_query,
                misleading_answer,
            )

            retrieval_attack_document = self.create_retrieval_attack_document(
                target_query,
            )

            combined_adversarial_document = f"""
                {retrieval_attack_document}

                {generator_attack_document}
            """

            adversarial_document_scenarios.append(combined_adversarial_document)

        return adversarial_document_scenarios

    def generate_malicious_corpus_for_target_queries(
        self,
        target_query_list: List[str],
    ) -> List[str]:
        """
        Generate a malicious corpus for multiple target queries.
        
        Args:
            target_query_list: List of queries to generate malicious documents for
            
        Returns:
            List of malicious document strings for all target queries
        """
        all_malicious_documents = []

        for single_query in target_query_list:
            all_malicious_documents.extend(self.generate_malicious_document_corpus(single_query))

        return all_malicious_documents
