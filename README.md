# Dynamic GNN-Based Runtime Jailbreak Detection

Project Overview
This project investigates the use of Dynamic Graph Neural Networks for runtime semantic jailbreak and prompt-injection detection in LLM pipelines.

The system represents LLM interactions as a dynamically updated, heterogeneous, provenance-aware graph.

A GNN analyses the graph to identify suspicious interaction patterns and a runtime policy engine decides whether proposed tool actions should be allowed, held for review, or blocked.

Core Modules
1. Dataset and Evaluation
2. Event Ingestion and Normalization
3. Dynamic Graph Construction
4. Feature Extraction and GNN Detection
5. Runtime Security Engine and API

Technology Stack
- Python
- PyTorch
- PyTorch Geometric
- Sentence Transformers
- NetworkX
- FastAPI
- Scikit-learn

Repository Structure
- configs/: Configuration
- data/: Datasets
- src/ingestion/: Event ingestion
- src/graph/: Dynamic graph construction
- src/features/: Feature extraction
- src/models/: Machine learning models
- src/runtime/: Runtime security
- src/evaluation/: Model evaluation
- api/: API endpoints
- dashboard/: Demonstration dashboard
- tests/: Automated tests
- docs/: Documentation
- notebooks/: Experiments

Development Status
Initial project structure created.