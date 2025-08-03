# Awesome RAG Attacks

A comprehensive framework for testing and evaluating attacks against Retrieval-Augmented Generation (RAG) systems, implementing various attack methodologies from the research literature.

## 🎯 Overview

This project implements a complete RAG attack research framework that includes:

- **Victim RAG System**: A simplified but functional RAG implementation using LangChain
- **Multiple RAG Attacks**: Implementation of various attack methodologies from the literature, including PoisonedRAG, CorruptRAG, and other knowledge corruption attacks
- **Evaluation Framework**: Comprehensive metrics for assessing attack effectiveness
- **Dataset Management**: Support for BEIR benchmark datasets (Natural Questions, MS MARCO, HotpotQA)

The framework is designed for security research and educational purposes to understand vulnerabilities in RAG systems and implement the growing body of research on RAG attacks.

## 🚧 Development Status

**This repository is under active development.** Our goal is to implement a comprehensive collection of RAG attacks from the research literature. We are continuously working to expand the framework with additional attack methodologies from published papers and improvements to existing implementations.

Current status:
- ✅ PoisonedRAG attack implementation (complete)
- ✅ CorruptRAG attack implementation (complete with its variations CorruptRAG-AS and CorruptRAG-AK)
- 🔄 Additional attack methods from literature (in progress)
- 🔄 Enhanced evaluation metrics (planned)
- 🔄 Defense mechanisms (planned)

**Available Attack Methods**:
- **PoisonedRAG**: A knowledge poisoning attack that generates malicious documents using both generator attacks (creating documents with incorrect information) and retrieval attacks (optimizing documents for high retrieval relevance while containing misleading content)
  - [PoisonedRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Generation](https://arxiv.org/abs/2402.07867)
- **CorruptRAG**: Template-based poisoning attacks with two variants:
  - **CorruptRAG-AS (Adversarial Suffix)**: Uses specific templates to construct poisoned text by combining target queries with adversarial templates claiming correct answers are outdated
  - **CorruptRAG-AK (Adversarial Knowledge)**: Builds on AS by using LLM refinement to make malicious documents more natural and coherent while preserving targeted misinformation
  - [Practical Poisoning Attacks against Retrieval-Augmented Generation](https://arxiv.org/abs/2504.03957)


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

### 2. RAG Attack Implementations (`src/attacks/`)

Implementation of various attacks against RAG systems from the research literature:

```python
from src.attacks.attack_factory import get_attack_class
from config.config import load_configuration

# Load configuration
config = load_configuration()

# Use attack factory to select attack method
attack = get_attack_class("poison_rag", config.attack_config)
# Or use corrupt rag attack
attack = get_attack_class("corrupt_rag", config.attack_config)

# Generate malicious documents for target queries
target_queries = ["What is the capital of France?"]
malicious_docs = attack.generate_malicious_corpus_for_target_queries(target_queries)

# Inject into RAG system
rag.insert_text(malicious_docs)
```

### 3. Dataset Management (`src/dataset_loader.py`)

Handles BEIR benchmark datasets:

```python
from src.dataset_loader import BeirDatasetLoader
from config.config import DatasetLoaderConfiguration

# Load dataset
config = DatasetLoaderConfiguration(
    dataset_name="nq",
    dataset_path="data/", 
    sample_size=100
)
loader = BeirDatasetLoader(config)
dataset = loader.load_beir_dataset()

# Convert to documents
documents = loader.create_documents_from_dataset(dataset)
```

**Supported Datasets**:
- **Natural Questions (nq)**: Real questions from Google search
- **MS MARCO (msmarco)**: Microsoft's reading comprehension dataset  
- **HotpotQA**: Multi-hop reasoning questions

## 🧪 Running Experiments

### Experiment 1: Basic Attack Effectiveness

```python
# Compare RAG responses before and after attack
from main import RagAttackOrchestrator
from src.victim_rag import VictimRAG
from src.dataset_loader import BeirDatasetLoader
from src.attacks.attack_factory import get_attack_class
from config.config import load_configuration

config = load_configuration()
rag = VictimRAG(configuration.rag_config)
dataset_loader = BeirDatasetLoader(configuration.dataset_loader_config)
attack = get_attack_class(attack_type, configuration.attack_config)

orchestrator = RagAttackOrchestrator(
    rag, 
    dataset_loader, 
    attack
)

# Setup RAG with benign documents
orchestrator.initialize_rag_system()

# Generate target queries
target_queries = orchestrator.benchmark_dataset.get_random_queries(num_queries=5)

# Get clean responses
clean_responses = orchestrator.victim_rag_system.answer_multiple_questions(target_queries)

# Poison the system
orchestrator.inject_malicious_documents(target_queries)

# Get poisoned responses  
poisoned_responses = orchestrator.victim_rag_system.answer_multiple_questions(target_queries)

# Compare results
for query, clean, poisoned in zip(target_queries, clean_responses, poisoned_responses):
    print(f"Query: {query}")
    print(f"Clean: {clean}")
    print(f"Poisoned: {poisoned}")
    print("---")
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
