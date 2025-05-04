# LLMOps Projects

This directory contains complete project examples for LLMOps.

## Directory Structure

- **FAST-API-LLM/**: FastAPI implementation for LLM serving
- **CICD-Medical-Chatbot/**: CI/CD pipeline for a medical chatbot
- **RAG-Application/**: Complete RAG application example

## Project Descriptions

### FAST-API-LLM

A FastAPI implementation for serving LLMs with the following features:
- RESTful API for text generation
- Model loading and caching
- Request validation and error handling
- Swagger documentation

**To run:**
```bash
cd FAST-API-LLM
pip install -r requirements.txt
uvicorn main:app --reload
```

### CICD-Medical-Chatbot

A medical chatbot with CI/CD pipeline that includes:
- GitHub Actions workflow for CI/CD
- Automated testing
- Docker containerization
- Deployment scripts

**To run:**
```bash
cd CICD-Medical-Chatbot
pip install -r requirements.txt
python app.py
```

### RAG-Application

A complete RAG application example with:
- Document loading and chunking
- Vector database integration
- Query processing
- Response generation

**To run:**
```bash
cd RAG-Application
pip install -r requirements.txt
python app.py
```

## Getting Started

Each project has its own README.md file with detailed instructions on how to set up and run the project. Navigate to the project directory and follow the instructions in the README.md file.
