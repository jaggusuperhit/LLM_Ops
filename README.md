# LLM-Ops

A comprehensive repository for Large Language Model Operations (LLMOps) containing tutorials, projects, demos, and documentation.

## Repository Structure

```
LLM-Ops/
├── Tutorials/
│   ├── MLflow/                  # MLflow tutorials for model tracking and management
│   ├── RAG/                     # Retrieval Augmented Generation tutorials
│   ├── Embeddings/              # Text embeddings and preprocessing tutorials
│   ├── LLM-Integration/         # LLM integration and usage tutorials
│   └── Vector-Databases/        # Vector database tutorials (FAISS, Chroma, Pinecone, Weaviate)
│
├── Projects/
│   ├── FAST-API-LLM/            # FastAPI implementation for LLM serving
│   ├── CICD-Medical-Chatbot/    # CI/CD pipeline for a medical chatbot
│   └── RAG-Application/         # Complete RAG application example
│
├── Demos/
│   ├── LangServe/               # LangChain serving demos
│   ├── LangSmith/               # LangChain monitoring and evaluation demos
│   └── Cloud-Integrations/      # Cloud provider integrations (AWS, GCP, Azure)
│
├── Documentation/               # Course materials and documentation
│
└── Utils/                       # Utility scripts and tools
```

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/LLM-Ops.git
   cd LLM-Ops
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv llmops
   # On Windows
   .\llmops\Scripts\activate
   # On Unix or MacOS
   source llmops/bin/activate
   ```

3. Install the required packages for the specific project or tutorial you want to run.

## Tutorials

The `Tutorials` directory contains Jupyter notebooks that demonstrate various aspects of LLMOps:

- **MLflow**: Learn how to track experiments, manage models, and deploy them using MLflow
- **RAG**: Explore Retrieval Augmented Generation techniques
- **Embeddings**: Understand text embeddings and preprocessing techniques
- **LLM-Integration**: Learn how to integrate and use LLMs in your applications
- **Vector-Databases**: Explore different vector databases for semantic search

## Projects

The `Projects` directory contains complete project examples:

- **FAST-API-LLM**: A FastAPI implementation for serving LLMs
- **CICD-Medical-Chatbot**: A medical chatbot with CI/CD pipeline
- **RAG-Application**: A complete RAG application example

## Demos

The `Demos` directory contains demonstration code:

- **LangServe**: Demos for LangChain serving
- **LangSmith**: Demos for LangChain monitoring and evaluation
- **Cloud-Integrations**: Demos for cloud provider integrations

## Documentation

The `Documentation` directory contains course materials and documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
