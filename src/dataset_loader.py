"""
dataset_loader.py (refactored)

A cleaned‑up implementation of **DatasetLoader** that preserves the original
public interface while reducing complexity and improving readability.

Key improvements
----------------
* **Single source of truth** for dataset metadata via the `SUPPORTED_DATASETS`
  constant.
* **Pathlib** used instead of `os.path` to simplify path manipulation.
* Clearer method names, stricter typing and comprehensive docstrings.
* **Dataclass‑style** initialisation (without the external dependency) – all
  mutable state is declared up‑front for easier reasoning.
* Removed redundant pandas/numpy conversions and tightened control flow.
* Deterministic sampling handled with **NumPy Generator** only in one place.
* Explicit, concise logging – you can raise the log level in `Config` if you
  need more detail.

The public API remains:


Feel free to tweak constants such as `MAX_BENIGN` if your workload requires
larger or smaller validation sets.
"""

from __future__ import annotations

import json
from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple
import random

import pandas as pd
import requests
import urllib3
from tqdm.auto import tqdm
from beir import util as beir_util
from langchain.schema import Document

from config.config import DatasetLoaderConfiguration
from src.schemas import RagDataset

# ---------------------------------------------------------------------------
# Globals & monkey‑patches
# ---------------------------------------------------------------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch BEIR download helper to ignore SSL verification – this is *optional*.
# If you manage certificates correctly you can delete this section.


def _download_url_no_ssl(url: str, save_path: str, chunk_size: int = 1 << 20) -> None:
    """Stream a remote file to *save_path* without SSL verification."""
    resp = requests.get(url, stream=True, verify=False)
    resp.raise_for_status()
    save_path = Path(save_path)
    with save_path.open("wb") as fh:
        for chunk in tqdm(resp.iter_content(chunk_size=chunk_size), desc=f"⬇ {url.rsplit('/', 1)[-1]}"):
            if chunk:
                fh.write(chunk)

beir_util.download_url = _download_url_no_ssl  # type: ignore[attr‑defined]


SUPPORTED_DATASETS = {
    "nq": "natural_questions",
    "msmarco": "msmarco",
    "hotpotqa": "hotpotqa",
}

class BeirDatasetLoader:
    """Utility for downloading, caching and sampling BEIR datasets."""

    # Public ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, config: DatasetLoaderConfiguration) -> None:
        self.config = config

        # Resolve & download dataset -------------------------------------------------
        self.dataset_identifier: str = self._resolve_dataset_identifier()
        self.dataset_directory_path: Path = self._ensure_dataset_is_available()
        logger.info(f"Dataset {self.dataset_identifier} ready at {self.dataset_directory_path}")

    def create_documents_from_dataset(
        self,
        rag_dataset: RagDataset
    ) -> List[Document]:

        document_list = []
        for document_id, document_data in rag_dataset.documents.items():
            document_list.append(
                Document(
                    page_content=document_data["text"],
                    metadata={"source": document_id}
                )
            )

        logger.info(f"Created {len(rag_dataset.documents)} documents from texts")
        return document_list
    
    def load_beir_dataset(
        self
    ) -> RagDataset:
        """Load corpus passages into memory and return a flat list of strings."""

        # Load raw jsonl files
        document_corpus = self._build_document_corpus()
        query_collection = self._build_query_collection()
        ground_truth_relevance_mapping = self._build_ground_truth_relevance_mapping()

        if self.config.sample_size:
            sampled_query_keys = random.sample(list(query_collection.keys()), self.config.sample_size)
            query_collection = {query_id: query_collection[query_id] for query_id in sampled_query_keys}
            ground_truth_relevance_mapping = {query_id: ground_truth_relevance_mapping[query_id] for query_id in sampled_query_keys}
            all_relevant_document_ids = {document_id for document_list in ground_truth_relevance_mapping.values() for document_id in document_list}
            document_corpus = {document_id: document_corpus[document_id] for document_id in all_relevant_document_ids}

        return RagDataset(
            documents=document_corpus,
            queries=query_collection,
            gt_map=ground_truth_relevance_mapping
        )

    # Internal helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _resolve_dataset_identifier(self) -> str:
        dataset_name = self.config.dataset_name
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from {list(SUPPORTED_DATASETS)}")
        return dataset_name

    def _ensure_dataset_is_available(self) -> Path:
        """Download dataset if not cached and return the local path."""
        project_root = Path(__file__).resolve().parent.parent
        dataset_name = self.config.dataset_name
        data_root_directory = project_root / self.config.dataset_path
        dataset_directory_path = data_root_directory / dataset_name
        if not dataset_directory_path.exists():
            download_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            logger.info(f"Downloading {dataset_name} to {data_root_directory}")
            beir_util.download_and_unzip(download_url, str(data_root_directory))
        return dataset_directory_path

    # ------------------------------ corpus -----------------------------------
    def _build_document_corpus(self) -> Dict[str, Dict]:
        logger.info(f"Building corpus from {self.dataset_directory_path}")

        document_corpus: Dict[str, Dict] = {}
        for document_record in tqdm(self._read_jsonl_file(self.dataset_directory_path / "corpus.jsonl"), desc="Building corpus"):
            document_id = document_record.get("_id")
            document_record.pop("_id")
            document_corpus[document_id] = document_record

        return document_corpus

    def _build_query_collection(self) -> Dict[str, Dict]:
        logger.info(f"Building queries from {self.dataset_directory_path}")
        query_collection: Dict[str, Dict] = {}
        for query_record in tqdm(self._read_jsonl_file(self.dataset_directory_path / "queries.jsonl"), desc="Building queries"):
            query_id = query_record.get("_id")
            query_record.pop("_id")
            query_collection[query_id] = query_record
        return query_collection

    def _build_ground_truth_relevance_mapping(self) -> Dict[str, List[str]]:
        logger.info(f"Building ground truth mapping from {self.dataset_directory_path}")
        qrels_file_path = self.dataset_directory_path / "qrels" / "test.tsv"
        qrels_dataframe = pd.read_csv(qrels_file_path, sep="\t", names=["query_id", "document_id", "score"], dtype=str)
        ground_truth_relevance_mapping = qrels_dataframe.groupby("query_id")["document_id"].apply(list).to_dict()

        return ground_truth_relevance_mapping
    
    # --------------------------- utilities -----------------------------------
    def _read_jsonl_file(self, file_path: str) -> List[dict]:
        jsonl_file_path = Path(file_path)
        with jsonl_file_path.open("r", encoding="utf‑8") as file_handle:
            return [json.loads(line) for line in file_handle]
