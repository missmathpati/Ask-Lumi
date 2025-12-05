"""
Backend module for LumiShop multimodal RAG system.

This module loads product data, embeddings, and CLIP models, and provides
functions for text and image-based product retrieval with LLM-generated answers.
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
from openai import OpenAI

# Get the root directory (where this file is located)
ROOT = Path(__file__).resolve().parent

# Data file paths - prefer parquet, fallback to CSV
PARQUET_PATH = ROOT / "amazon_processed.parquet"
CSV_PATH = ROOT / "amazon_processed.csv"
EMB_PATH = ROOT / "embeddings_cache.pkl"
HEAD_PATH = ROOT / "clip_head.pth"  # Optional: saved fine-tuned head

# Global variables to hold loaded data and models (loaded once)
products_df: Optional[pd.DataFrame] = None
item_embs: Optional[np.ndarray] = None
base_model: Optional[CLIPModel] = None
processor: Optional[CLIPProcessor] = None
head: Optional[torch.nn.Module] = None
openai_client: Optional[OpenAI] = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Track what's been loaded
models_loaded = False
data_loaded = False


def _load_products() -> pd.DataFrame:
    """
    Load product data from parquet (preferred) or CSV (fallback).
    
    Returns:
        DataFrame with product information
    """
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(
            f"Neither {PARQUET_PATH.name} nor {CSV_PATH.name} found in {ROOT}"
        )
    
    return df


def _load_embeddings() -> np.ndarray:
    """
    Load item embeddings from pickle file.
    
    Assumptions about embeddings_cache.pkl structure:
    - If it's a dict, it may have keys like "embeddings", "combined_embs", "item_embs", etc.
    - If it's a numpy array directly, use it as-is
    - Embeddings should be 2D array of shape (n_products, embedding_dim)
    - Embeddings should be L2-normalized for cosine similarity
    
    Returns:
        2D numpy array of shape (n_products, embedding_dim)
    """
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Embeddings file {EMB_PATH.name} not found in {ROOT}")
    
    with open(EMB_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different possible structures
    if isinstance(data, dict):
        # Try common key names (prioritize item_embs as that's what the notebook uses)
        if "item_embs" in data:
            embs = data["item_embs"]
        elif "embeddings" in data:
            embs = data["embeddings"]
        elif "combined_embs" in data:
            embs = data["combined_embs"]
        else:
            # Use the first array-like value found
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    embs = value
                    break
            else:
                raise ValueError(
                    f"Could not find embeddings array in pickle dict. "
                    f"Available keys: {list(data.keys())}"
                )
    elif isinstance(data, np.ndarray):
        embs = data
    else:
        raise ValueError(
            f"Unexpected embeddings format. Expected dict or numpy array, got {type(data)}"
        )
    
    # Ensure it's a 2D array
    if embs.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embs.shape}")
    
    # L2 normalize to ensure cosine similarity works correctly
    # Note: embeddings should already be normalized, but we normalize again to be safe
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-12, None)
    
    print(f"Loaded embeddings: shape {embs.shape}, dtype {embs.dtype}")
    
    return embs.astype(np.float32)


def _load_models():
    """Load CLIP model, processor, and fine-tuned head (if available)."""
    global base_model, processor, head, models_loaded
    
    if models_loaded:
        return
    
    model_name = "openai/clip-vit-base-patch32"
    
    # Load CLIP base model and processor
    print("Loading CLIP model...")
    base_model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    base_model.eval()
    
    # Try to load fine-tuned head if available
    if HEAD_PATH.exists():
        try:
            print(f"Loading fine-tuned head from {HEAD_PATH.name}...")
            from torch import nn
            
            embed_dim = base_model.config.projection_dim  # 512 for this model
            
            class ClipHead(nn.Module):
                def __init__(self, embed_dim):
                    super().__init__()
                    self.image_proj = nn.Linear(embed_dim, embed_dim, bias=False)
                    self.text_proj = nn.Linear(embed_dim, embed_dim, bias=False)
                    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
                
                def forward(self, image_features, text_features):
                    img = self.image_proj(image_features)
                    txt = self.text_proj(text_features)
                    img = img / img.norm(dim=-1, keepdim=True)
                    txt = txt / txt.norm(dim=-1, keepdim=True)
                    logit_scale = self.logit_scale.exp()
                    logits_per_image = logit_scale * img @ txt.t()
                    logits_per_text = logits_per_image.t()
                    return logits_per_image, logits_per_text, img, txt
            
            head = ClipHead(embed_dim).to(device)
            head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
            head.eval()
            print("Fine-tuned head loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned head: {e}")
            print("⚠️  WARNING: Using base CLIP features for queries.")
            print("   If your item_embs are from fine-tuned CLIP, results may be poor.")
            print("   To fix: Save the head from your notebook: torch.save(head.state_dict(), 'clip_head.pth')")
            head = None
    else:
        print("⚠️  WARNING: No fine-tuned head found (clip_head.pth).")
        print("   Using base CLIP features for queries.")
        print("   If your item_embs are from fine-tuned CLIP, results may be poor.")
        print("   To fix: In your notebook, after training, run: torch.save(head.state_dict(), 'clip_head.pth')")
        head = None
    
    models_loaded = True


def _load_openai_client():
    """Initialize OpenAI client."""
    global openai_client
    
    if openai_client is None:
        # Try multiple methods to get API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Try loading from .env file if available
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
            except ImportError:
                pass  # python-dotenv not installed, skip
        
        # If still no key, try to read from a config file
        if not api_key:
            config_file = ROOT / ".openai_key"
            if config_file.exists():
                try:
                    api_key = config_file.read_text().strip()
                except Exception:
                    pass
        
        # Last resort: check if it was set in the environment by the notebook
        # (This handles the case where the key might be in os.environ but not getenv)
        if not api_key and "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it using one of these methods:\n"
                "1. Environment variable: export OPENAI_API_KEY='your-key'\n"
                "2. Create a .env file with: OPENAI_API_KEY=your-key\n"
                "3. Create a .openai_key file in the project root with your key\n"
                "4. Set it in your shell before running: OPENAI_API_KEY=your-key streamlit run app.py"
            )
        
        # Initialize client - OpenAI() will also check the environment automatically
        openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()


def _ensure_data_loaded():
    """Ensure products and embeddings are loaded (lazy loading)."""
    global products_df, item_embs, data_loaded
    
    if not data_loaded:
        products_df = _load_products()
        item_embs = _load_embeddings()
        data_loaded = True
        
        # Verify dimensions match
        if len(products_df) != len(item_embs):
            raise ValueError(
                f"Mismatch: {len(products_df)} products but {len(item_embs)} embeddings"
            )
        
        # Check if image_path column exists and show sample
        if "image_path" in products_df.columns:
            sample_paths = products_df["image_path"].dropna().head(3).tolist()
            print(f"Sample image paths: {sample_paths}")
        else:
            print("⚠️  Warning: 'image_path' column not found in products dataframe")


def encode_query_text(query: str) -> np.ndarray:
    """
    Encode a text query using CLIP (with fine-tuned head if available).
    
    Args:
        query: Text query string
    
    Returns:
        Normalized query embedding vector
    """
    _load_models()
    
    inputs = processor(
        text=[query],
        images=None,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        txt_feat = base_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        if head is not None:
            # Use fine-tuned head
            _, _, _, txt_proj = head(
                image_features=torch.zeros_like(txt_feat),  # dummy, not used
                text_features=txt_feat
            )
            q = txt_proj[0].cpu().numpy()
        else:
            # Use base CLIP features
            q = txt_feat[0].cpu().numpy()
    
    # L2 normalize
    q = q / np.linalg.norm(q)
    return q


def encode_query_image(image: Image.Image) -> np.ndarray:
    """
    Encode an image using CLIP (with fine-tuned head if available).
    
    Args:
        image: PIL Image object
    
    Returns:
        Normalized query embedding vector
    """
    _load_models()
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = processor(
        text=None,
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        img_feat = base_model.get_image_features(pixel_values=inputs["pixel_values"])
        
        if head is not None:
            # Use fine-tuned head
            _, _, img_proj, _ = head(
                image_features=img_feat,
                text_features=torch.zeros_like(img_feat)  # dummy
            )
            q = img_proj[0].cpu().numpy()
        else:
            # Use base CLIP features
            q = img_feat[0].cpu().numpy()
    
    # L2 normalize
    q = q / np.linalg.norm(q)
    return q


def retrieve_products_from_query_vec(q_vec: np.ndarray, k: int = 5) -> pd.DataFrame:
    """
    Retrieve top-k products based on query embedding.
    
    Args:
        q_vec: Normalized query embedding vector (d,)
        k: Number of products to retrieve
    
    Returns:
        DataFrame with top-k products, sorted by similarity (descending)
    """
    _ensure_data_loaded()
    
    # Ensure q_vec is normalized
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # Ensure dimensions match
    if q_vec.shape[0] != item_embs.shape[1]:
        raise ValueError(
            f"Query embedding dimension ({q_vec.shape[0]}) doesn't match "
            f"item embeddings dimension ({item_embs.shape[1]})"
        )
    
    # Compute cosine similarities (both are normalized, so dot product = cosine similarity)
    sims = item_embs @ q_vec  # (N,)
    
    # Get top-k indices
    topk_idx = np.argsort(-sims)[:k]
    
    # Get corresponding products
    results = products_df.iloc[topk_idx].copy()
    results["similarity_score"] = sims[topk_idx]
    
    print(f"Retrieved {len(results)} products. Similarity scores: {sims[topk_idx]}")
    
    return results


def format_rag_context(products: pd.DataFrame, max_items: int = 5) -> str:
    """
    Format retrieved products into a context string for the LLM.
    
    Args:
        products: DataFrame with retrieved products
        max_items: Maximum number of products to include
    
    Returns:
        Formatted context string
    """
    rows = []
    for i, (_, row) in enumerate(products.head(max_items).iterrows(), start=1):
        name = row.get("Product Name", "")
        brand = row.get("Brand Name", "")
        category = row.get("Category", row.get("Main Category", ""))
        price = row.get("Selling Price", "")
        desc = row.get("product_text", "")
        
        snippet = desc[:600] + ("..." if len(desc) > 600 else "") if desc else ""
        
        rows.append(
            f"Product {i}:\n"
            f"  Name: {name}\n"
            f"  Brand: {brand}\n"
            f"  Category: {category}\n"
            f"  Price: {price}\n"
            f"  Details: {snippet}\n"
        )
    return "\n".join(rows)


def build_rag_prompt(user_query: str, products: pd.DataFrame) -> str:
    """
    Build RAG prompt for LLM with user query and retrieved products.
    
    Args:
        user_query: User's question
        products: DataFrame with retrieved products
    
    Returns:
        Formatted prompt string
    """
    context_block = format_rag_context(products)
    
    prompt = f"""You are a helpful e-commerce assistant working over a fixed product catalog.
You can ONLY use the retrieved product information below.

Instructions:
- First, check if any retrieved product clearly matches what the user is asking about.
- If none of the products seem to match well, say that the catalog does not contain
  that exact product and that you are only suggesting similar alternatives.
- Do NOT invent brand or product names that are not present in the retrieved context.
- When recommending products, mention their names and key features explicitly.
- Be concise, factual, and cautious when you are unsure.

Customer query:
{user_query}

Retrieved products:
{context_block}

Now answer the customer's question using ONLY the products above.
"""
    return prompt


def generate_answer(prompt: str) -> str:
    """
    Generate answer using OpenAI's GPT model.
    
    Args:
        prompt: RAG prompt string
    
    Returns:
        Generated answer string
    """
    _load_openai_client()
    
    try:
        # Use standard OpenAI Chat Completions API
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini (you can change to gpt-4o or gpt-4-turbo)
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        # Fallback if API call fails
        raise RuntimeError(f"Failed to generate answer from OpenAI: {e}")


def answer_text_query_rag(query: str, k: int = 5) -> Tuple[str, pd.DataFrame]:
    """
    Given a text query, run retrieval over the embeddings and return:
      - answer: a natural language answer string
      - retrieved_df: a DataFrame with the top-k retrieved products
    
    Args:
        query: User's text query
        k: Number of products to retrieve
    
    Returns:
        Tuple of (answer_string, retrieved_products_dataframe)
    """
    _ensure_data_loaded()
    _load_models()
    
    # 1. Encode query
    q_vec = encode_query_text(query)
    
    # 2. Retrieve products
    retrieved = retrieve_products_from_query_vec(q_vec, k=k)
    
    # 3. Build RAG prompt
    prompt = build_rag_prompt(query, retrieved)
    
    # 4. Generate answer
    answer = generate_answer(prompt)
    
    return answer, retrieved


def answer_image_query_rag(image_bytes: bytes, query: Optional[str], k: int = 5) -> Tuple[str, pd.DataFrame]:
    """
    Given an image (and optional text query), run multimodal retrieval and return:
      - answer: a natural language answer string
      - retrieved_df: a DataFrame with the top-k retrieved products
    
    Args:
        image_bytes: Image file as bytes
        query: Optional text query to accompany the image
        k: Number of products to retrieve
    
    Returns:
        Tuple of (answer_string, retrieved_products_dataframe)
    """
    _ensure_data_loaded()
    _load_models()
    
    # Default query if none provided
    if query is None or query.strip() == "":
        query = "Identify and describe this product and how it is used."
    
    # 1. Load image from bytes
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # 2. Encode image
    q_vec = encode_query_image(image)
    
    # 3. Retrieve products
    retrieved = retrieve_products_from_query_vec(q_vec, k=k)
    
    # 4. Build RAG prompt
    prompt = build_rag_prompt(query, retrieved)
    
    # 5. Generate answer
    answer = generate_answer(prompt)
    
    return answer, retrieved
