"""
Victim RAG System Implementation

This module implements a simplified Retrieval-Augmented Generation (RAG) system 
using LangChain components. The system is designed as a "victim" for testing 
RAG attack scenarios, particularly PoisonedRAG attacks.

The RAG system follows a standard pipeline:
1. Document preprocessing and embedding
2. Vector storage using FAISS
3. Question answering with retrieval-augmented prompts
4. Support for document insertion (for attack scenarios)

Key features:
- Uses modern LangChain components for flexibility
- FAISS vector storage for efficient similarity search
- Configurable language models and embeddings
- Support for batch querying and document retrieval
- Context-based querying similar to PoisonedRAG approach

This implementation prioritizes simplicity and compatibility with attack research
over production-level optimizations.
"""

import os
from typing import List, Dict, Any, Optional
import warnings
import requests
from huggingface_hub import configure_http_backend

from langchain_community.vectorstores import FAISS
from langchain.chat_models.base import init_chat_model
from langchain.embeddings.base import init_embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from loguru import logger

from config.config import RagSystemConfiguration
from src.prompts import RAG_QA_TEMPLATE


# Suppress all relevant warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Set environment variable to disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backend_factory)


class VictimRAG:
    """
    A simplified RAG implementation designed for attack research.
    
    This class implements a Retrieval-Augmented Generation system using LangChain
    components. It's designed to be compatible with PoisonedRAG attack scenarios
    while maintaining simplicity and flexibility.
    
    The system processes documents without chunking (following PoisonedRAG approach)
    and supports dynamic document insertion for attack simulation.
    
    Attributes:
        config: Configuration containing model settings and parameters
        language_model: The language model for generating responses
        vector_database: FAISS vector store for document similarity search
        retrieval_chain: LangChain retrieval QA chain
        indexed_documents: List of processed documents
        embedding_model: Embedding model for document vectorization
    """

    def __init__(self, config: RagSystemConfiguration):
        """
        Initialize the RAG system with provided configuration.
        
        Args:
            config: RagConfig object containing all necessary settings including
                embedding model, chat model, and processing parameters
        """
        self.config = config
        self.language_model = None
        self.vector_database = None
        self.retrieval_chain = None
        self.indexed_documents = []
        self.embedding_model = None
        
        # Setup components
        self._initialize_language_model()
        self._initialize_embedding_model()
        
    def _initialize_language_model(self):
        """
        Setup the language model based on configuration.
        
        Initializes the language model using LangChain's init_chat_model
        with parameters from the configuration. Supports various providers
        like OpenAI, Anthropic, etc.
        
        Raises:
            Exception: If LLM initialization fails
        """
        chat_configuration = self.config.chat_config

        try:
            self.language_model = init_chat_model(**chat_configuration.model_dump())
            logger.info(
                f"LLM initialized with parameters: {chat_configuration.model_dump()}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _initialize_embedding_model(self):
        """
        Setup the embeddings model based on configuration.
        
        Initializes the embedding model using LangChain's init_embeddings
        with parameters from the configuration. Supports HuggingFace 
        sentence transformers and other embedding providers.
        
        Raises:
            Exception: If embeddings initialization fails
        """
        embedding_configuration = self.config.embedding_config.model_dump()

        try:
            self.embedding_model = init_embeddings(**embedding_configuration)
            logger.info(
                f"Embeddings initialized with parameters: {embedding_configuration}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def prepare_documents(self, documents: List[Document]):
        """
        Prepare documents for indexing without chunking.
        
        Following the PoisonedRAG approach, documents are processed as complete 
        units rather than being split into chunks. This maintains document 
        coherence and is better suited for attack scenarios.
        
        The method performs basic preprocessing:
        - Normalizes whitespace
        - Filters out empty documents  
        - Combines title with content if available
        - Truncation is handled by the embedding model's tokenizer
        
        Args:
            documents: List of LangChain Document objects to process
            
        Returns:
            List of processed Document objects ready for vectorization
        """
        # Simple preprocessing: normalize whitespace and ensure non-empty content
        processed_documents = []
        for document in documents:
            content = document.page_content.strip()
            if content:  # Only keep non-empty documents
                # Combine title and content if title exists in metadata
                if document.metadata.get('title'):
                    content = f"{document.metadata['title']} {content}"
                
                processed_document = Document(
                    page_content=content,
                    metadata=document.metadata
                )
                processed_documents.append(processed_document)
        
        logger.info(f"Processed {len(processed_documents)} documents")
        max_token_length = self.config.processing_params.max_length
        logger.info(
            f"Documents will be truncated at {max_token_length} tokens by embedding model"
        )

        self.indexed_documents = processed_documents
        return processed_documents

    def build_vectorstore(self, documents: List[Document]):
        """
        Build FAISS vectorstore from processed documents.
        
        Creates a FAISS vector index from the provided documents using
        the configured embedding model. FAISS provides efficient similarity
        search capabilities for document retrieval.
        
        Args:
            documents: List of processed Document objects to index
        """
        logger.info("Building vectorstore...")

        self.vector_database = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        logger.info("Vectorstore built successfully")

    def insert_documents(self, documents: List[Document]):
        """
        Insert additional documents into the existing vectorstore.
        
        This method allows dynamic addition of documents to the vector index,
        which is useful for attack scenarios where malicious documents need
        to be injected into the system.
        
        Args:
            documents: List of Document objects to add to the vectorstore
        """
        self.vector_database.add_documents(documents)
        logger.info(f"Inserted {len(documents)} documents into the vectorstore")

    def insert_text(
            self,
            text_strings: List[str],
            metadata_list: Optional[List[Dict]] = None
        ):
        """
        Insert raw text strings into the vectorstore.
        
        This method provides a convenient way to add text content directly
        without creating Document objects first. Particularly useful for
        attack scenarios where malicious text needs to be injected.
        
        Args:
            text_strings: List of text strings to add to the vectorstore
            metadata_list: Optional list of metadata dictionaries for each text
        """
        self.vector_database.add_texts(text_strings, metadata=metadata_list)
        logger.info(f"Inserted {len(text_strings)} text into the vectorstore")

    def setup_retrieval_chain(self):
        """
        Setup the retrieval QA chain for question answering.
        
        Creates a LangChain RetrievalQA chain that combines document retrieval
        with language model generation. Uses a custom prompt template similar
        to PoisonedRAG for consistency with attack research.
        
        The chain performs:
        1. Retrieval of top-k relevant documents
        2. Context formatting with custom prompt
        3. Answer generation using the language model
        
        Raises:
            ValueError: If vectorstore is not built first
        """
        if not self.vector_database:
            raise ValueError(
                "Vectorstore not built. Call build_vectorstore first."
            )
        
        # Create custom prompt template similar to PoisonedRAG
        prompt = PromptTemplate(
            template=RAG_QA_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Create retriever
        processing_parameters = self.config.processing_params
        document_retriever = self.vector_database.as_retriever(
            search_kwargs={"k": processing_parameters.top_k}
        )

        # Create retrieval chain
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.language_model,
            chain_type="stuff",
            retriever=document_retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        logger.info("Retrieval chain setup complete")

    def execute_retrieval_query(self, question: str) -> Dict[str, Any]:
        """
        Execute a query against the RAG system and return detailed results.
        
        Args:
            question: The question to query the RAG system with
            
        Returns:
            Dictionary containing the question, answer, and source documents
        """
        if not self.retrieval_chain:
            raise ValueError(
                "Retrieval chain not setup. Call setup_retrieval_chain first."
            )

        result = self.retrieval_chain.invoke({"query": question})

        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": source_document.page_content,
                    "metadata": source_document.metadata
                }
                for source_document in result["source_documents"]
            ]
        }

        return response

    def answer_question(self, question: str) -> str:
        """
        Query the RAG system and return just the answer.
        
        Args:
            question: The question to query the RAG system with
            
        Returns:
            The answer string from the RAG system
        """
        response = self.execute_retrieval_query(question)
        return response["answer"]

    def get_retrieved_documents(self, question: str) -> List[str]:
        """
        Get relevant documents for a question.
        
        Args:
            question: The question to retrieve documents for
            
        Returns:
            List of document content strings that are relevant to the question
        """
        if not self.vector_database:
            raise ValueError("Vectorstore not built. Call build_vectorstore first.")

        response = self.execute_retrieval_query(question)
        source_documents = response["source_documents"]
        return [document["content"] for document in source_documents]
    
    def get_retrieved_documents_for_queries(self, queries: List[str]) -> List[List[str]]:
        """
        Get relevant documents for a list of queries.
        
        Args:
            queries: List of questions to retrieve documents for
            
        Returns:
            List of lists, where each inner list contains document content strings
            relevant to the corresponding query
        """
        return [self.get_retrieved_documents(query) for query in queries]

    def answer_multiple_questions(self, questions: List[str]) -> List[str]:
        """
        Query the RAG system for a list of questions.
        
        Args:
            questions: List of questions to query the RAG system with
            
        Returns:
            List of answer strings corresponding to each question
        """
        return [self.answer_question(question) for question in questions]

    def answer_question_with_context(self, question: str, context_strings: List[str]) -> str:
        """Query with provided contexts (similar to PoisonedRAG approach)."""
        combined_context = "\n".join(context_strings)

        formatted_prompt = RAG_QA_TEMPLATE.format(
            context=combined_context, question=question
        )

        try:
            # Use invoke method for all LangChain versions
            result = self.language_model.invoke(formatted_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            logger.error(f"Error in answer_question_with_context: {e}")
            response = f"Error processing query: {str(e)}"

        return response

    def save_vectorstore(self, file_path: str):
        """Save the vectorstore to disk."""
        if self.vector_database:
            self.vector_database.save_local(file_path)
            logger.info(f"Vectorstore saved to {file_path}")
    
    def load_vectorstore(self, file_path: str):
        """Load vectorstore from disk."""
        if not self.embedding_model:
            raise ValueError(
                "Embeddings not initialized. Cannot load vectorstore."
            )

        self.vector_database = FAISS.load_local(
            file_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vectorstore loaded from {file_path}")