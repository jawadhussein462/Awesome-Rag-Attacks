# Awesome RAG Attacks

A comprehensive framework for testing and evaluating attacks against Retrieval-Augmented Generation (RAG) systems, implementing various attack methodologies from the research literature.

## 🎯 Overview

This project implements a complete RAG attack research framework that includes:

- **Victim RAG System**: A simplified but functional RAG implementation using LangChain
- **Multiple RAG Attacks**: Implementation of various attack methodologies from the literature, including PoisonedRAG and other knowledge corruption attacks
- **Evaluation Framework**: Comprehensive metrics for assessing attack effectiveness
- **Dataset Management**: Support for BEIR benchmark datasets (Natural Questions, MS MARCO, HotpotQA)

The framework is designed for security research and educational purposes to understand vulnerabilities in RAG systems and implement the growing body of research on RAG attacks.

## 🚧 Development Status

**This repository is under active development.** Our goal is to implement a comprehensive collection of RAG attacks from the research literature. We are continuously working to expand the framework with additional attack methodologies from published papers and improvements to existing implementations.

Current status:
- ✅ PoisonedRAG attack implementation (complete)
- ✅ CorruptRAG attack implementation (complete with it's variations CorruptRAG-AS and CorruptRAG-AK))
- 🔄 Additional attack methods from literature (in progress)
- 🔄 Enhanced evaluation metrics (planned)
- 🔄 Defense mechanisms (planned)

## 📁 Project Structure

```
awesome-rag-attacks/
├── src/                              # Main source code
│   ├── victim_rag.py                 # RAG system implementation  
│   ├── attacks/                      # Attack implementations
│   │   ├── poisoned_rag_attack.py    # PoisonedRAG attack
│   │   ├── corrupt_rag_attack.py     # Corrupt RAG attack
│   │   └── attack_factory.py         # Attack selection factory
│   ├── dataset_loader.py         # BEIR dataset handling
│   ├── evaluation.py             # Evaluation metrics
│   ├── schemas.py                # Data structures
│   └── prompts.py                # Prompt templates
├── config/                       # Configuration files
│   ├── config.py                 # Configuration classes
│   └── config.yaml               # Default settings
├── tests/                        # Test files
│   └── testing_rag.py            # RAG system tests
├── main.py                       # Main orchestrator
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```


## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for language models)
- 8GB+ RAM (for dataset processing)
- Internet connection (for dataset downloads)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd awesome-rag-attacks
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

#### Running the Complete Attack Pipeline

```bash
python main.py
```

You can also specify which attack to use:
```bash
python main.py --attack poisoned_rag  # Use PoisonedRAG attack
python main.py --attack corrupt_rag   # Use Corrupt RAG attack
```

This will:
1. Load and sample a dataset (Natural Questions by default)
2. Build a RAG system with the benign documents
3. Generate target queries for attack
4. Create malicious documents using the selected attack method
5. Insert malicious documents into the RAG system
6. Compare responses before and after poisoning

#### Configuration

Edit `config/config.yaml` to customize your configurations

## 🔧 Components

### 1. Victim RAG System (`src/victim_rag.py`)

A LangChain-based RAG implementation designed for attack research:

```python
from src.victim_rag import VictimRAG
from config.config import RagConfig, load_settings

# Initialize RAG system
config = load_settings().rag_config
rag = VictimRAG(config)

# Load and process documents
documents = [Document(page_content="Your content here", metadata={})]
processed_docs = rag.prepare_documents(documents)
rag.build_vectorstore(processed_docs)
rag.setup_retrieval_chain()

# Query the system
answer = rag.query("What is machine learning?")
print(answer)
```

**Key Features**:
- No document chunking (following PoisonedRAG methodology)
- FAISS vector storage for efficient retrieval
- Support for dynamic document insertion
- Configurable embedding and language models
- Batch querying capabilities

### 2. RAG Attack Implementations (`src/attacks/`)

Implementation of various attacks against RAG systems from the research literature:

```python
from src.attack_factory import get_attack

# Use attack factory to select attack method
attack = get_attack("poisoned_rag", attack_config)
# Or use corrupt rag attack
attack = get_attack("corrupt_rag", attack_config)

# Generate malicious documents for target queries
target_queries = ["What is the capital of France?"]
malicious_docs = attack.generate_malicious_corpus_for_target_queries(target_queries)

# Inject into RAG system
rag.insert_text(malicious_docs)
```

**Available Attack Methods**:
- **PoisonedRAG**: Knowledge poisoning through malicious document injection
- **Corrupt RAG**: Document corruption attack method


### 3. Dataset Management (`src/dataset_loader.py`)

Handles BEIR benchmark datasets:

```python
from src.dataset_loader import DatasetLoader
from config.config import DatasetLoaderConfig

# Load dataset
config = DatasetLoaderConfig(
    dataset_name="nq",
    dataset_path="data/", 
    sample_size=100
)
loader = DatasetLoader(config)
dataset = loader.load_dataset()

# Convert to documents
documents = loader.load_documents_from_dataset(dataset)
```

**Supported Datasets**:
- **Natural Questions (nq)**: Real questions from Google search
- **MS MARCO (msmarco)**: Microsoft's reading comprehension dataset  
- **HotpotQA**: Multi-hop reasoning questions

### 4. Evaluation Framework (`src/evaluation.py`)

Comprehensive metrics for attack assessment:

```python
from src.evaluation import Evaluator

evaluator = Evaluator()

# Evaluate retrieval performance
recalls, precisions = evaluator.evaluate_retrieval_metrics(
    retrieved_docs, relevant_docs
)

print(f"Average Recall: {recalls:.3f}")
print(f"Average Precision: {precisions:.3f}")
```

## 🧪 Running Experiments

### Experiment 1: Basic Attack Effectiveness

```python
# Compare RAG responses before and after attack
from main import Orchestrator
from config.config import load_settings

config = load_settings()
orchestrator = Orchestrator(
    config.rag_config, 
    config.dataset_loader_config, 
    config.poisoned_rag_attack_config
)

# Setup RAG with benign documents
orchestrator.setup_rag()

# Generate target queries
target_queries = orchestrator.rag_dataset.get_random_queries(num_queries=5)

# Get clean responses
clean_responses = orchestrator.rag.query_list_of_questions(target_queries)

# Poison the system
orchestrator.poison_rag(target_queries)

# Get poisoned responses  
poisoned_responses = orchestrator.rag.query_list_of_questions(target_queries)

# Compare results
for query, clean, poisoned in zip(target_queries, clean_responses, poisoned_responses):
    print(f"Query: {query}")
    print(f"Clean: {clean}")
    print(f"Poisoned: {poisoned}")
    print("---")
```

### Experiment 2: Retrieval Quality Analysis

```python
# Analyze how attacks affect retrieval quality
recalls_before, precisions_before = orchestrator.evaluate_rag()
print(f"Before Attack - Recall: {recalls_before:.3f}, Precision: {precisions_before:.3f}")

orchestrator.poison_rag(target_queries)

recalls_after, precisions_after = orchestrator.evaluate_rag()  
print(f"After Attack - Recall: {recalls_after:.3f}, Precision: {precisions_after:.3f}")
```

## 📊 Evaluation Metrics

The framework provides several evaluation metrics:

- **Retrieval Recall**: Fraction of relevant documents retrieved
- **Retrieval Precision**: Fraction of retrieved documents that are relevant
- **Attack Success Rate**: Percentage of queries returning incorrect answers
- **Response Quality**: Semantic similarity between clean and poisoned responses

## ⚙️ Advanced Configuration

### Custom Models

```yaml
rag_config:
  embedding_config:
    model: "sentence-transformers/all-mpnet-base-v2"  # Better but slower
    provider: "huggingface"
  chat_config:
    model: "gpt-4"  # More capable but expensive
    model_provider: "openai"
```

### Attack Parameters

```yaml
poisoned_rag_attack_config:
  num_docs_per_target_query: 10      # More attack documents
  num_target_queries: 5              # Fewer queries to focus on
  num_words_per_doc: 50              # Longer malicious documents
  seed: 42                           # Reproducible results
```

### Dataset Sampling

```yaml
dataset_loader_config:
  dataset_name: "hotpotqa"           # Multi-hop reasoning
  sample_size: 500                   # Larger dataset
```

## 📋 Requirements

Key dependencies:
- `langchain>=0.2.14` - RAG framework
- `langchain-community>=0.2.14` - Community components
- `langchain-openai>=0.1.25` - OpenAI integration
- `faiss-cpu>=1.8.0` - Vector similarity search
- `sentence-transformers>=3.0.1` - Text embeddings
- `pandas>=2.2.2` - Data manipulation
- `loguru>=0.7.2` - Logging

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Priority areas for development:

- **RAG Attack Methods**: Implementation of attacks from recent research papers
- **Defense Mechanisms**: Methods to detect and prevent various attacks  
- **Evaluation Metrics**: More sophisticated assessment methods for different attack types
- **Model Support**: Integration with additional LLM providers
- **Performance**: Optimization for larger datasets

We especially welcome implementations of new attack methods from the literature. Please ensure any new attacks include proper attribution to the original research.

## ⚠️ Ethical Considerations

This framework is designed for:
- ✅ Security research and education
- ✅ Understanding RAG vulnerabilities  
- ✅ Developing defense mechanisms
- ✅ Academic research

**NOT for**:
- ❌ Attacking production systems without permission
- ❌ Spreading misinformation
- ❌ Malicious activities

## 📄 License

MIT License - see LICENSE file for details.

## 📚 References

This framework implements attacks and methodologies from various research papers:

**Primary Attack References**:
- [PoisonedRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Generation](https://arxiv.org/abs/2402.07867)
- [Practical Poisoning Attacks against Retrieval-Augmented Generation](https://arxiv.org/abs/2504.03957)

**Dataset and Evaluation References**:
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)

*Additional references will be added as more attack methods are implemented from the literature.*

## 🆘 Support

For issues and questions:
1. Check existing GitHub issues
2. Review configuration documentation
3. Ensure proper API key setup
4. Verify dataset downloads completed

For bugs or feature requests, please open an issue with:
- System information (OS, Python version)
- Error messages or logs
- Steps to reproduce
- Expected vs actual behavior
