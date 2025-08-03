"""
CorruptRAG-AS and CorruptRAG-AK Attack Implementation

This module implements both variants of CorruptRAG attacks against 
Retrieval-Augmented Generation (RAG) systems as described in the research 
paper.

CorruptRAG-AS (Adversarial Suffix):
The attack uses a specific template-based approach where poisoned text is 
constructed using the formula:
pi = qi ⊕ phi_adv ⊕ phi_state

Where:
- qi: target query
- phi_adv: adversarial template claiming correct answers are outdated
- phi_state: statement template presenting the targeted answer as current
- ⊕: concatenation operator

CorruptRAG-AK (Adversarial Knowledge):
Builds upon AS by adding a refinement step where an LLM processes the
AS-generated text to make it more natural and coherent while preserving
the targeted misinformation. This addresses the generalizability limitations
of the AS approach.

Key advantages:
1. Single Document: Only one malicious document needed per target query
2. Template-based (AS): Uses specific, tested templates for effectiveness
3. Natural Language (AK): Refined text appears more legitimate
4. Higher Success Rate: More effective at manipulating RAG responses
5. Practical Feasibility: Simple to implement and deploy

Based on the research paper: "CorruptRAG: Practical Poisoning Attacks 
against Retrieval-Augmented Generation"
"""

from typing import List

from config.config import CorruptRAGAttackConfiguration
from langchain.chat_models import init_chat_model
from src.prompts import (
    CORRUPT_RAG_AS_CORRECT_ANSWER_PROMPT,
    CORRUPT_RAG_AS_MISLEADING_ANSWER_PROMPT,
    CORRUPT_RAG_AS_PHI_ADV_TEMPLATE,
    CORRUPT_RAG_AS_PHI_STATE_TEMPLATE,
    CORRUPT_RAG_AK_REFINEMENT_PROMPT,
    CORRUPT_RAG_AK_VALIDATION_PROMPT
)
from loguru import logger

class CorruptRAG:
    """
    Implements CorruptRAG-AS and CorruptRAG-AK attacks against RAG systems.

    This class generates poisoned text using either:
    - CorruptRAG-AS: Template-based approach (qi ⊕ phi_adv ⊕ phi_state)
    - CorruptRAG-AK: AS approach + LLM refinement for natural text

    Attributes:
        attack_configuration: Configuration for the attack parameters
        adversarial_llm_config: Configuration for the language model used
        adversarial_language_model: The LLM instance for generating content
    """

    def __init__(
        self,
        corrupt_rag_attack_config: CorruptRAGAttackConfiguration
    ):
        """
        Initialize the CorruptRAG-AS attack system.
        
        Args:
            corrupt_rag_attack_config: Configuration containing attack 
                parameters, including LLM settings.
        """
        self.attack_configuration = corrupt_rag_attack_config
        self.adversarial_llm_config = (
            self.attack_configuration.llm_attack_config
        )
        self.adversarial_language_model = init_chat_model(
            **self.adversarial_llm_config.model_dump()
        )

    def get_correct_answer(self, target_query: str) -> str:
        """
        Get the correct answer for a target query.
        
        This method generates what would be considered the correct answer
        for the given query, which will be used in the phi_adv template
        to claim it's outdated.
        
        Args:
            target_query: The query to get the correct answer for
            
        Returns:
            The correct answer for the query
        """
        prompt = CORRUPT_RAG_AS_CORRECT_ANSWER_PROMPT.format(target_query=target_query)
        response = self.adversarial_language_model.invoke(prompt)
        return response.content.strip()

    def create_misleading_answer(self, target_query: str) -> str:
        """
        Generate an incorrect answer for a target query.
        
        This method creates the targeted answer (Ai) that the attacker
        wants the RAG system to produce.
        
        Args:
            target_query: The query for which to generate an incorrect answer
            
        Returns:
            A string containing an incorrect but plausible answer
        """
        prompt = CORRUPT_RAG_AS_MISLEADING_ANSWER_PROMPT.format(target_query=target_query)
        response = self.adversarial_language_model.invoke(prompt)
        return response.content.strip()

    def create_phi_adv(self, correct_answer: str) -> str:
        """
        Create the adversarial sub-template phi_adv.
        
        This template claims that the correct answer is outdated, exploiting
        LLMs' tendency to trust recent information.
        
        Args:
            correct_answer: The correct answer to claim is outdated
            
        Returns:
            The phi_adv template string
        """
        return CORRUPT_RAG_AS_PHI_ADV_TEMPLATE.format(correct_answer=correct_answer)

    def create_phi_state(self, targeted_answer: str) -> str:
        """
        Create the statement sub-template phi_state.
        
        This template presents the targeted (incorrect) answer as confirmed
        by recent data.
        
        Args:
            targeted_answer: The incorrect answer to present as correct
            
        Returns:
            The phi_state template string
        """
        return CORRUPT_RAG_AS_PHI_STATE_TEMPLATE.format(targeted_answer=targeted_answer)

    def create_poisoned_text(
        self,
        target_query: str,
        correct_answer: str,
        targeted_answer: str
    ) -> str:
        """
        Create poisoned text using the CorruptRAG-AS formula.
        
        Implements: pi = qi ⊕ phi_adv ⊕ phi_state
        
        Args:
            target_query: The target query (qi)
            correct_answer: The actual correct answer (Ci)
            targeted_answer: The desired incorrect answer (Ai)
            
        Returns:
            Complete poisoned text string
        """
        # Create the sub-templates
        phi_adv = self.create_phi_adv(correct_answer)
        phi_state = self.create_phi_state(targeted_answer)
        
        # Concatenate according to the formula: pi = qi ⊕ phi_adv ⊕ phi_state
        poisoned_text = f"{target_query} {phi_adv} {phi_state}"
        
        return poisoned_text

    def refine_poisoned_text_ak(
        self,
        original_poisoned_text: str,
        target_query: str,
        targeted_answer: str
    ) -> str:
        """
        Refine poisoned text using CorruptRAG-AK approach.
        
        This method takes the AS-generated poisoned text and refines it
        using few-shot learning to make it more natural and coherent,
        with validation to ensure the targeted answer is preserved.
        
        Args:
            original_poisoned_text: The AS-generated poisoned text
            target_query: The target query
            targeted_answer: The desired incorrect answer
            
        Returns:
            Refined poisoned text that maintains effectiveness
        """
        max_attempts = self.attack_configuration.max_refinement_attempts
        v_words = self.attack_configuration.ak_word_limit
        
        for attempt in range(max_attempts):
            # Step 1: Refine the poisoned text
            refinement_prompt = CORRUPT_RAG_AK_REFINEMENT_PROMPT.format(
                original_text=original_poisoned_text, 
                target_query=target_query, 
                v_words=v_words
            )
            
            refined_response = self.adversarial_language_model.invoke(
                refinement_prompt
            )
            refined_text = refined_response.content.strip()
            
            # Step 2: Validate that refined text still produces targeted answer
            validation_prompt = CORRUPT_RAG_AK_VALIDATION_PROMPT.format(
                target_query=target_query,
                refined_text=refined_text
            )
        
            validation_response = self.adversarial_language_model.invoke(
                validation_prompt
            )
            validation_answer = validation_response.content.strip().lower()

            # Check if the validation matches our targeted answer
            # Simple containment check - could be made more sophisticated
            if targeted_answer.lower() in validation_answer:
                return refined_text

        # If we reach here, refinement failed - return original AS text
        return refined_text

    def generate_malicious_document_for_query(self, target_query: str) -> str:
        """
        Generate a malicious document using CorruptRAG-AS/AK methods.
        
        This is the main method that implements the complete CorruptRAG attack
        pipeline for a single query. It supports both AS and AK methods based
        on the configuration.
        
        Args:
            target_query: The query to generate a malicious document for
            
        Returns:
            Poisoned text string designed to mislead the RAG system
        """
        # Step 1: Get the correct answer (Ci)
        correct_answer = self.get_correct_answer(target_query)
        
        # Step 2: Generate the targeted incorrect answer (Ai)  
        targeted_answer = self.create_misleading_answer(target_query)
        
        # Step 3: Create poisoned text using the CorruptRAG-AS formula
        poisoned_text = self.create_poisoned_text(
            target_query,
            correct_answer,
            targeted_answer
        )
        
        # Step 4: If using AK method, refine the poisoned text
        if self.attack_configuration.attack_method == "AK":
            poisoned_text = self.refine_poisoned_text_ak(
                poisoned_text,
                target_query,
                targeted_answer
            )
        
        return poisoned_text

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
            List of poisoned text strings, one per target query
        """
        all_poisoned_texts = []

        for single_query in target_query_list:
            poisoned_text = self.generate_malicious_document_for_query(
                single_query
            )
            all_poisoned_texts.append(poisoned_text)

        return all_poisoned_texts 