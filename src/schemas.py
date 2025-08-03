"""
Data Schemas for RAG Attack Framework

This module defines the core data structures used throughout the RAG attack framework.
The main schema is RagDataset, which provides a standardized interface for working
with BEIR benchmark datasets and their associated queries and ground truth mappings.

The schemas are designed to be compatible with both the victim RAG system and
the attack evaluation pipeline, providing consistent data access patterns
across different components.
"""

from dataclasses import dataclass
from typing import Dict, List
import random


@dataclass
class RagDataset:
    documents: Dict[str, Dict]
    queries: Dict[str, Dict]
    gt_map: Dict[str, List[str]]  # keeping original name for compatibility

    def get_queries_text(self) -> List[str]:
        """Get all query texts from the dataset."""
        query_texts = [query_data["text"] for query_data in self.queries.values()]
        return query_texts

    def get_random_queries(self, seed: int = 42, num_queries: int = 1) -> List[str]:
        """Get a random sample of queries from the dataset."""
        random.seed(seed)
        return random.sample(self.get_queries_text(), num_queries)

    def get_relevant_documents(self, query: str) -> List[str]:
        """Get relevant documents for a given query."""
        query_id = None
        for current_query_id, query_data in self.queries.items():
            if query_data["text"] == query:
                query_id = current_query_id
                break

        if query_id is None:
            raise ValueError(f"Query not found: {query}")

        document_ids = self.gt_map[query_id]
        documents = [self.documents[doc_id] for doc_id in document_ids]
        document_texts = [doc["text"] for doc in documents]
        return document_texts
    
    def get_relevant_documents_for_queries(self, queries: List[str]) -> List[List[str]]:
        """Get relevant documents for a list of queries."""
        return [self.get_relevant_documents(query) for query in queries]
