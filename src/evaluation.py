"""
Evaluation Metrics for RAG Attack Assessment

This module implements evaluation metrics for assessing the effectiveness of attacks
against RAG systems. It provides standard information retrieval metrics including
precision and recall, which are essential for understanding how attacks affect
the quality of document retrieval.

The evaluation framework supports:
- Retrieval quality assessment (precision, recall)
- Attack effectiveness measurement  
- Before/after attack comparisons
- Batch evaluation for multiple queries

These metrics are crucial for quantifying the impact of poisoning attacks on
RAG system performance and for developing effective defense mechanisms.
"""

from typing import List
import numpy as np
from tqdm import tqdm


class RetrievalEvaluator:

    def calculate_retrieval_recall(
        self,
        retrieved_document_list: List[str],
        relevant_document_ids: List[str],
    ) -> float:
        """
        Calculate retrieval recall for a single query.

        Args:
            retrieved_document_list: List of retrieved documents with metadata
            relevant_document_ids: Set of relevant document IDs from ground truth  

        Returns:
            Recall score (0.0 to 1.0)
        """
        if len(retrieved_document_list) == 0:
            return 0.0

        retrieved_document_set = set(retrieved_document_list)
        relevant_document_set = set(relevant_document_ids)

        relevant_documents_retrieved = retrieved_document_set.intersection(relevant_document_set)

        recall_score = len(relevant_documents_retrieved) / len(relevant_document_set)
        return recall_score

    def calculate_retrieval_precision(
        self,
        retrieved_document_list: List[str],
        relevant_document_ids: List[str],
    ) -> float:
        """
        Calculate retrieval precision for a single query.
        
        Args:
            retrieved_document_list: List of retrieved document IDs
            relevant_document_ids: List of relevant document IDs from ground truth
            
        Returns:
            Precision score (0.0 to 1.0)
        """
        if len(retrieved_document_list) == 0:
            return 0.0

        retrieved_document_set = set(retrieved_document_list)
        relevant_document_set = set(relevant_document_ids)

        precision_score = len(retrieved_document_set.intersection(relevant_document_set)) / len(retrieved_document_set)
        return precision_score

    def evaluate_retrieval_metrics(
        self,
        retrieved_document_lists: List[List[str]],
        relevant_document_id_lists: List[List[str]],
    ) -> float:
        """
        Evaluate retrieval recall for a list of queries.
        """
        if len(retrieved_document_lists) != len(relevant_document_id_lists):
            raise ValueError("Number of retrieved and relevant documents must match")

        recall_scores = []
        precision_scores = []

        for retrieved_documents, relevant_documents in tqdm(zip(retrieved_document_lists, relevant_document_id_lists)):
            recall_score = self.calculate_retrieval_recall(retrieved_documents, relevant_documents)
            precision_score = self.calculate_retrieval_precision(retrieved_documents, relevant_documents)
            recall_scores.append(recall_score)
            precision_scores.append(precision_score)

        average_recall = np.mean(recall_scores)
        average_precision = np.mean(precision_scores)

        return average_recall, average_precision
