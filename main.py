"""
Main Orchestrator for RAG Attack Framework

This module provides the main orchestration logic for running RAG attack experiments.
The Orchestrator class coordinates all components of the framework:

1. Dataset loading and preprocessing
2. RAG system initialization and setup
3. Attack generation and execution
4. Evaluation and results collection

The main() function demonstrates a complete attack pipeline from start to finish,
including before/after comparisons to assess attack effectiveness.

Usage:
    python main.py

The script will:
- Load a sample dataset (configurable)
- Build a victim RAG system
- Generate target queries for attack
- Create and inject malicious documents
- Compare responses before and after poisoning

Configuration is handled through config/config.yaml and can be customized
for different datasets, models, and attack parameters.
"""

from typing import List, Optional
import argparse
import os

from loguru import logger
from langchain.schema import Document

from config.config import load_configuration
from src.victim_rag import VictimRAG
from src.dataset_loader import BeirDatasetLoader
from src.evaluation import RetrievalEvaluator
from src.attacks.attack_factory import get_attack_class
from src.schemas import RagDataset
from src.attacks.base_attack import BaseRAGAttack

class RagAttackOrchestrator:
    """
    Main orchestrator for RAG attack experiments.
    
    This class coordinates all components of the RAG attack framework,
    providing a high-level interface for running complete attack pipelines.
    It manages the lifecycle of experiments including dataset loading, RAG
    system setup, attack generation, and evaluation.
    
    The orchestrator follows a typical experimental workflow:
    1. Initialize all components with their respective configurations
    2. Load and prepare datasets for the experiment
    3. Set up the victim RAG system with benign documents  
    4. Execute attacks by generating and injecting malicious content
    5. Evaluate the impact on system performance
    
    Attributes:
        rag_config: Configuration for the victim RAG system
        dataset_loader_config: Configuration for dataset loading
        attack_config: Configuration for attack parameters
        attack_type: String identifier for the attack type
        victim_rag_system: Instance of the victim RAG system
        dataset_loader: Instance of the dataset loader
        evaluator: Instance of the evaluation framework
        attack_system: Instance of the attack system
        benchmark_dataset: Loaded dataset with queries and documents
        corpus_documents: Processed documents for the RAG system
    """

    def __init__(
        self,
        rag: VictimRAG,
        dataset_loader: BeirDatasetLoader,
        attack: BaseRAGAttack,
        evaluator: Optional[RetrievalEvaluator] = None,
    ):

        self.victim_rag_system = rag
        self.dataset_loader = dataset_loader
        self.evaluator = evaluator
        self.attack_system = attack

        self.benchmark_dataset: RagDataset = None
        self.corpus_documents: List[Document] = None

    def initialize_rag_system(self):

        # Load dataset
        self.benchmark_dataset: RagDataset = (
            self.dataset_loader.load_beir_dataset()
        )
        self.corpus_documents = (
            self.dataset_loader.create_documents_from_dataset(
                self.benchmark_dataset
            )
        )

        # Prepare documents
        processed_documents = self.victim_rag_system.prepare_documents(
            self.corpus_documents
        )

        # Setup vectorstore and retrieval chain
        self.victim_rag_system.build_vectorstore(processed_documents)
        self.victim_rag_system.setup_retrieval_chain()

    def inject_malicious_documents(self, target_queries: List[str]):
        malicious_document_corpus = (
            self.attack_system
            .generate_malicious_corpus_for_target_queries(target_queries)
        )
        self.victim_rag_system.insert_text(malicious_document_corpus)
        self.victim_rag_system.setup_retrieval_chain()

    def compute_retrieval_metrics(self):

        query_texts = self.benchmark_dataset.get_queries_text()
        retrieved_document_lists = self.victim_rag_system.get_retrieved_documents_for_queries(query_texts)
        relevant_document_lists = self.benchmark_dataset.get_relevant_documents_for_queries(query_texts)
        recall_scores, precision_scores = self.evaluator.evaluate_retrieval_metrics(retrieved_document_lists, relevant_document_lists)

        return recall_scores, precision_scores


def main(attack_type: str = "poison_rag"):
    """
    Main function to run RAG attack experiments.

    Args:
        attack_type: Type of attack to run. Options: "poison_rag", "corrupt_rag"
    """
    # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

    configuration = load_configuration()

    rag = VictimRAG(configuration.rag_config)
    dataset_loader = BeirDatasetLoader(configuration.dataset_loader_config)
    attack = get_attack_class(attack_type, configuration.attack_config)
    evaluator = RetrievalEvaluator()

    orchestrator = RagAttackOrchestrator(
        rag, dataset_loader, attack, evaluator
    )
    orchestrator.initialize_rag_system()
    target_queries = orchestrator.benchmark_dataset.get_random_queries(
        seed=attack.seed,
        num_queries=attack.num_target_queries
    )

    logger.info(f"Target queries: {target_queries}")
    logger.info(f"Response before poisoning: "
                f"{orchestrator.victim_rag_system.answer_multiple_questions(target_queries)}")

    orchestrator.inject_malicious_documents(target_queries)

    logger.info(f"Response after poisoning: "
                f"{orchestrator.victim_rag_system.answer_multiple_questions(target_queries)}")

    # recall_scores, precision_scores = orchestrator.compute_retrieval_metrics()
    # logger.info(f"Recalls: {recall_scores}")
    # logger.info(f"Precisions: {precision_scores}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG attack experiments"
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default="corrupt_rag",
        choices=["poison_rag", "corrupt_rag"],
        help="Type of attack to run (default: poison_rag)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.attack_type)
