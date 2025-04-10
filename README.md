# RAG-Optimization

**Optimizing Retrieval-Augmented Generation with information-heoretic approaches and end-to-end training (under development).**

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)

## Overview

RAG-Optimization is my goal to create a comprehensive framework for improving Retrieval-Augmented Generation (RAG) systems through information-theoretic approaches and novel end-to-end training capabilities. It aims to addresses key challenge in RAG systems of optimizing the organization of knowledge while leveraging state-of-the-art training optimizations like PEFT, mixed precision training, 8-bit optimizer compatibility etc.
Specifically, I aim to solve the intractable optimization of selecting the datastore contents and optimal model (generator) parameters under memory and compute constraints by using information-theoretic approaches. This repository contains a modular codebase that aims to enable researchers and developers to build, train, and evaluate such optimal RAG systems.

## Key Features

- **Independently Developed**: 3k+ lines of code written from scratch, covering the full pipeline from document processing to evaluation and visualization
- **Information-Theoretic Document Selection**: Smart scoring mechanisms to strategically select documents for the datastore, achieving up to 30% improvement in accuracy per unit of memory
- **End-to-End Optimization**: Fine-tuning framework for decoder-only LLMs that enables gradient flow to the retriever component (uses [ContextFlow](https://github.com/shubhampatel77/contextflow))
- **Comprehensive Evaluation**: Robust metrics for evaluating RAG systems including factuality, relevance, and faithfulness
- **Memory-Efficient Training**: Support for LoRA adapters, mixed precision, 8-bit quantization, and gradient checkpointing
- **Advanced Document Processing**: Temporal scoring, recency analysis, relevance assessment, and document ranking
- **Visualization Tools**: Interactive visualizations for performance analysis and data exploration

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate

### Install from source
```bash
git clone https://github.com/shubhampatel77/rag-optimization.git
cd rag-optimization
pip install -e .
```

## Architecture

RAG-Optimization consists of several key components:

- **Document Selection & Indexing**: Intelligently selects and indexes documents based on information-theoretic scoring, including temporal relevance, uniqueness, and complexity metrics.
- **Dense Passage Retriever (DPR)**: Customized implementation integrated with FAISS for fast approximate nearest neighbor search.
- **Training Pipeline**: End-to-end optimization with custom loss functions that allow gradient flow from answer generation back to retrieval.
- **Evaluation Framework**: Comprehensive metrics for assessing RAG system performance across different dimensions.
- **Visualization Tools**: Components for analyzing and visualizing system performance.

## Documentation

### Key Modules
- `process_documents.py`: Document scoring and selection tools
- `update_retriever.py`: Retriever component update functions
- `update_generator.py`: Generator model update and training
- `rag.py`: Core RAG implementation
- `dataloader.py`: Data loading utilities
- `evals.py`: Evaluation metrics and functions
- `visualization.py`: Performance visualization tools

## Project Status

This is an active research project with ongoing development. Major features are stable, but improvements and new capabilities are continuously being added.

### Roadmap
- [ ] Add usage and detailed examples
- [ ] Add detailed documentation and split code with ContextFlow
- [ ] Additional scoring mechanisms for document selection
- [ ] Support for multi-modal document types
- [ ] Improved long-context handling
- [ ] Enhanced evaluation metrics
- [ ] Performance optimizations and ablations
- [ ] LLM feedback + RL to enhance scoring functions
- [ ] Theoretical formalism and uncertainty bounds 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
