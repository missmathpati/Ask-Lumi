# Lumi 
# Multimodal Product Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) system for e-commerce product search and recommendation, featuring both text and image-based query capabilities.

## Overview

This application demonstrates a complete multimodal RAG pipeline that enables users to search and discover products using natural language text queries or by uploading product images. The system leverages fine-tuned CLIP embeddings for semantic search, FAISS for efficient similarity retrieval, and OpenAI GPT for generating natural language responses.

## Features

- **Text-Based Search**: Query products using natural language (e.g., "wireless bluetooth headphones under $50")
- **Image-Based Search**: Upload product images to find visually similar items
- **Multimodal RAG**: Combines retrieval and generation for intelligent product recommendations
- **Interactive UI**: Clean, user-friendly Streamlit interface with product cards and detailed information
- **Fine-Tuned Models**: CLIP model fine-tuned on Amazon product dataset for improved retrieval accuracy

## Architecture

The system is built with a modular architecture:

- **Models Module** (`models.py`): Handles CLIP model loading, fine-tuned head initialization, and FAISS index management
- **Retrieval Module** (`retrieval.py`): Implements text and image query encoding and product retrieval
- **RAG Pipeline Module** (`rag_pipeline.py`): Formats retrieved context and generates answers using OpenAI GPT
- **Streamlit Application** (`app.py`): User interface for interacting with the system

## Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Ensure you have the required data files:
   - `amazon_processed.csv`: Processed product dataset
   - `clip_head.pth`: Fine-tuned CLIP head weights
   - `images/`: Directory containing product images

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. The application will open in your browser. You can:
   - Use the **Text Query** tab to search products by text
   - Use the **Image Query** tab to upload images and find similar products

## Technical Details

### Models
- **Base Model**: OpenAI CLIP (ViT-Base/32)
- **Fine-Tuning**: Lightweight projection head trained on Amazon product dataset
- **Embeddings**: Combined text and image embeddings (512-dimensional)
- **Index**: FAISS IndexFlatIP for cosine similarity search

### Dataset
- Amazon product dataset with 9,970 products
- Each product includes: name, brand, category, price, description, and image
- Products span multiple categories: Toys & Games, Home & Kitchen, Electronics, etc.

### Performance
- Recall@1: 33.3%
- Recall@5: 60.9%
- Recall@10: 71.4%

## Project Structure

```
.
├── app.py                 # Streamlit application
├── models.py              # Model loading and initialization
├── retrieval.py           # Product retrieval functions
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Python dependencies
├── amazon_processed.csv   # Processed product dataset
├── clip_head.pth         # Fine-tuned CLIP head weights
├── embeddings_cache.pkl  # Cached embeddings (auto-generated)
├── images/               # Product images directory
└── explanation.md        # Detailed project documentation
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- FAISS
- OpenAI API access

See `requirements.txt` for complete dependency list.

## Notes

- First run will compute embeddings and may take several minutes
- Subsequent runs use cached embeddings for faster startup
- Ensure sufficient disk space for images and embeddings cache
- OpenAI API usage will incur costs based on API calls


This project is developed for academic purposes as part of a course project.

