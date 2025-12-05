"""
Model loading and initialization module.
Handles loading CLIP model, fine-tuned head, and FAISS index.
"""

import torch
import torch.nn as nn
import numpy as np
import faiss
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import os


class ClipHead(nn.Module):
    """Fine-tuned CLIP projection head."""
    
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


class ModelLoader:
    """Handles loading and initialization of models and data."""
    
    def __init__(self, device=None, progress_callback=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "openai/clip-vit-base-patch32"
        self.base_model = None
        self.head = None
        self.processor = None
        self.df = None
        self.index = None
        self.item_embs = None
        self.progress_callback = progress_callback  # Streamlit progress callback
        
    def _log(self, message):
        """Log message using callback if available, otherwise print."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
        
    def load_models(self):
        """Load CLIP base model and fine-tuned head."""
        self._log("Loading CLIP base model...")
        self.base_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.base_model.eval()
        
        self._log("Loading CLIP processor...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        self._log("Initializing fine-tuned head...")
        embed_dim = self.base_model.config.projection_dim
        self.head = ClipHead(embed_dim).to(self.device)
        
        head_path = "clip_head.pth"
        if os.path.exists(head_path):
            self._log(f"Loading fine-tuned head weights from {head_path}...")
            self.head.load_state_dict(torch.load(head_path, map_location=self.device))
        else:
            self._log(f"Warning: {head_path} not found. Using untrained head.")
        
        self.head.eval()
        self._log("✓ Models loaded successfully!")
        
    def load_data(self, csv_path="amazon_processed.csv"):
        """Load product dataframe and create image_path if needed."""
        self._log(f"Loading product data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        initial_count = len(self.df)
        
        # Check what image data we have
        has_image_url = "Image" in self.df.columns and self.df["Image"].notna().any()
        has_image_path_col = "image_path" in self.df.columns and self.df["image_path"].notna().any()
        images_dir_exists = os.path.exists("images") and os.path.isdir("images")
        
        # Determine strategy: prioritize local files if images directory exists
        if images_dir_exists or has_image_path_col:
            # Use local image files
            if "image_path" not in self.df.columns:
                self._log("Creating image_path column from Uniq Id...")
                def get_image_path(row):
                    """Generate local image path from Uniq Id or index."""
                    uniq_id = row.get("Uniq Id", None)
                    if pd.isna(uniq_id) or uniq_id is None:
                        uniq_id = row.name  # fallback to index
                    # Convert to string and ensure it's clean
                    uniq_id_str = str(uniq_id).strip()
                    return f"images/{uniq_id_str}.jpg"
                
                self.df["image_path"] = self.df.apply(get_image_path, axis=1)
                self._log("✓ Created image_path column")
        
        # Filter to rows with images (either Image URL or image_path exists)
        has_image_path = "image_path" in self.df.columns and self.df["image_path"].notna().any()
        
        if has_image_path:
            # Filter to rows where image_path exists and file exists locally
            self._log("Filtering to products with local images...")
            self.df = self.df[self.df["image_path"].notna()].copy()
            self._log(f"  Products with image_path: {len(self.df)}")
            
            # Check which image files actually exist
            existing_mask = self.df["image_path"].apply(lambda p: os.path.exists(p) if pd.notna(p) else False)
            self.df = self.df[existing_mask].reset_index(drop=True)
            
            if len(self.df) == 0:
                # Check if images directory exists
                images_dir = "images"
                if not os.path.exists(images_dir):
                    # Provide helpful error with solution
                    error_msg = (
                        f"❌ No products found with existing image files!\n\n"
                        f"Problem: The 'images/' directory doesn't exist.\n\n"
                        f"Your CSV has {initial_count} products with Image URLs, but local image files are required.\n\n"
                        f"Solution Options:\n"
                        f"1. Download images:\n"
                        f"   - Create an 'images/' directory\n"
                        f"   - Download product images from the 'Image' URLs in your CSV\n"
                        f"   - Save them as: images/{{Uniq Id}}.jpg\n\n"
                        f"2. Quick test (use sample data):\n"
                        f"   - Create 'images/' directory\n"
                        f"   - Add at least a few product images matching the Uniq Id format\n\n"
                        f"3. Check your CSV:\n"
                        f"   - Ensure 'Uniq Id' column exists\n"
                        f"   - Verify image filenames match Uniq Id values\n\n"
                    )
                    if has_image_url:
                        error_msg += (
                            f"\n💡 Tip: You have Image URLs in your CSV. You can download images using:\n"
                            f"   - A script to download from URLs\n"
                            f"   - Or manually download a subset for testing"
                        )
                    raise ValueError(error_msg)
                else:
                    # Count files in images directory
                    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    # Get a sample image_path before filtering (need to restore df temporarily)
                    sample_df_before = pd.read_csv(csv_path)
                    if "image_path" not in sample_df_before.columns:
                        sample_df_before["image_path"] = sample_df_before.apply(
                            lambda row: f"images/{str(row.get('Uniq Id', row.name)).strip()}.jpg", axis=1
                        )
                    sample_path = sample_df_before['image_path'].iloc[0] if len(sample_df_before) > 0 else 'N/A'
                    raise ValueError(
                        f"❌ No products found with existing image files!\n\n"
                        f"Problem: {len(image_files)} image files found in 'images/' directory, "
                        f"but none match the image_path values in the CSV.\n\n"
                        f"Possible issues:\n"
                        f"1. Image filenames don't match 'Uniq Id' values from CSV\n"
                        f"2. Image paths in CSV are incorrect\n"
                        f"3. File extensions don't match (.jpg vs .png)\n\n"
                        f"Initial products in CSV: {initial_count}\n"
                        f"Image files in 'images/': {len(image_files)}\n"
                        f"Sample image_path expected: {sample_path}\n"
                        f"Sample files in images/: {image_files[:5] if image_files else 'None'}\n\n"
                        f"💡 Tip: Image files should be named exactly as: images/{{Uniq Id}}.jpg"
                    )
            
            self._log(f"✓ Found {len(self.df)} products with existing image files")
        elif has_image_url:
            # If we have Image URLs but no local files, warn but continue
            self._log("Warning: Image URLs found but no local image files detected.")
            self._log("Filtering to products with Image URLs...")
            self.df = self.df[self.df["Image"].notna()].reset_index(drop=True)
            
            if len(self.df) == 0:
                raise ValueError(
                    "❌ No products found with Image URLs!\n\n"
                    "The CSV has an 'Image' column but all values are empty or invalid."
                )
            
            self._log(f"✓ Loaded {len(self.df)} products with Image URLs")
            self._log("Note: Images will need to be downloaded or image_path created for full functionality")
        else:
            raise ValueError(
                "❌ No image data found in CSV!\n\n"
                "The CSV file must have either:\n"
                "1. An 'Image' column with image URLs, OR\n"
                "2. An 'image_path' column with local file paths\n\n"
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Check for product_text column
        if "product_text" not in self.df.columns:
            self._log("⚠️  Warning: 'product_text' column not found. Creating from available columns...")
            # Try to create product_text from available columns
            text_cols = ["Product Name", "Brand Name", "Category", "Description", "About Product"]
            available_cols = [col for col in text_cols if col in self.df.columns]
            if available_cols:
                self.df["product_text"] = self.df[available_cols].fillna("").agg(" ".join, axis=1)
                self._log(f"✓ Created 'product_text' from: {', '.join(available_cols)}")
            else:
                raise ValueError(
                    f"❌ Cannot create 'product_text' column!\n\n"
                    f"None of the expected text columns found: {text_cols}\n"
                    f"Available columns: {list(self.df.columns)}"
                )
        
        # Check that product_text has valid data
        if self.df["product_text"].isna().all() or (self.df["product_text"].str.strip() == "").all():
            raise ValueError(
                "❌ All 'product_text' values are empty!\n\n"
                "The product_text column exists but contains no valid text data."
            )
        
        self._log(f"✓ Loaded {len(self.df)} products")
        
    def load_or_compute_embeddings(self, embeddings_cache="embeddings_cache.pkl"):
        """
        Load embeddings from cache or compute them.
        Also builds FAISS index.
        """
        import pickle
        
        if os.path.exists(embeddings_cache):
            self._log(f"Loading embeddings from cache...")
            try:
                with open(embeddings_cache, "rb") as f:
                    cache_data = pickle.load(f)
                    # Handle different cache formats
                    if isinstance(cache_data, dict) and "item_embs" in cache_data:
                        self.item_embs = cache_data["item_embs"]
                    elif isinstance(cache_data, np.ndarray):
                        self.item_embs = cache_data
                    else:
                        raise ValueError("Unknown cache format")
                    self._log(f"✓ Loaded embeddings: {self.item_embs.shape}")
            except Exception as e:
                self._log(f"Error loading cache: {e}. Recomputing embeddings...")
                self.item_embs = self._compute_embeddings()
                self._log(f"✓ Computed embeddings: {self.item_embs.shape}")
                self._log(f"Saving to cache...")
                with open(embeddings_cache, "wb") as f:
                    pickle.dump({"item_embs": self.item_embs}, f)
                self._log("✓ Cache saved")
        else:
            self._log("=" * 60)
            self._log("FIRST-TIME SETUP: Computing embeddings...")
            self._log("This will take 5-10 minutes but only happens once!")
            self._log("=" * 60)
            self.item_embs = self._compute_embeddings()
            self._log(f"✓ Computed embeddings: {self.item_embs.shape}")
            self._log(f"Saving to cache for faster future loads...")
            with open(embeddings_cache, "wb") as f:
                pickle.dump({"item_embs": self.item_embs}, f)
            self._log("✓ Cache saved - future loads will be much faster!")
        
        # Build FAISS index
        self._log("Building FAISS index...")
        d = self.item_embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.item_embs.astype("float32"))
        self._log(f"✓ FAISS index built with {self.index.ntotal} vectors")
        
    def _compute_embeddings(self, batch_size=64):
        """Compute combined text+image embeddings for all products."""
        # Ensure we have products to process
        if self.df is None or len(self.df) == 0:
            raise ValueError(
                "No products found in dataframe. Check that:\n"
                "1. The CSV file exists and has data\n"
                "2. Products have valid image paths\n"
                "3. Image files exist in the 'images/' directory"
            )
        
        # Ensure we have the required columns
        if "product_text" not in self.df.columns:
            raise ValueError("Missing 'product_text' column in dataframe")
        
        if "image_path" not in self.df.columns:
            raise ValueError("Missing 'image_path' column. Run load_data() first.")
        
        texts = self.df["product_text"].tolist()
        paths = self.df["image_path"].tolist()
        
        all_text_embs = []
        all_image_embs = []
        
        total_batches = (len(self.df) + batch_size - 1) // batch_size
        self._log(f"Processing {total_batches} batches of {batch_size} products each...")
        self._log(f"Total products to process: {len(self.df)}")
        
        # Use tqdm for progress in terminal
        iterator = tqdm(range(0, len(self.df), batch_size), desc="Encoding products", unit="batch")
        
        for batch_idx, start in enumerate(iterator):
            end = min(start + batch_size, len(self.df))
            batch_texts = texts[start:end]
            batch_paths = paths[start:end]
            
            # Load images
            images = []
            for p in batch_paths:
                if pd.isna(p) or not isinstance(p, str):
                    # Use blank image if path is invalid
                    img = Image.new("RGB", (224, 224), (255, 255, 255))
                else:
                    try:
                        if os.path.exists(p):
                            img = Image.open(p).convert("RGB")
                        else:
                            # File doesn't exist, use blank image
                            img = Image.new("RGB", (224, 224), (255, 255, 255))
                    except Exception:
                        # Any error loading image, use blank
                        img = Image.new("RGB", (224, 224), (255, 255, 255))
                images.append(img)
            
            # Process
            inputs = self.processor(
                text=batch_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                img_feat = self.base_model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
                txt_feat = self.base_model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                _, _, img_proj, txt_proj = self.head(img_feat, txt_feat)
            
            all_image_embs.append(img_proj.cpu().numpy())
            all_text_embs.append(txt_proj.cpu().numpy())
        
        # Validate that we processed at least one batch
        if len(all_image_embs) == 0 or len(all_text_embs) == 0:
            raise ValueError(
                "No embeddings were computed. This usually means:\n"
                "1. No products were found after filtering\n"
                "2. The dataframe is empty\n"
                f"Current dataframe length: {len(self.df)}\n"
                "Please check:\n"
                "- That 'amazon_processed.csv' has data\n"
                "- That products have valid 'product_text' column\n"
                "- That image paths exist (check 'images/' directory)"
            )
        
        self._log("Combining and normalizing embeddings...")
        ft_image_embs = np.vstack(all_image_embs)
        ft_text_embs = np.vstack(all_text_embs)
        
        # Combine and normalize
        combined = (ft_text_embs + ft_image_embs) / 2.0
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / np.clip(norms, 1e-12, None)
        
        return combined.astype("float32")
    
    def initialize_all(self, csv_path="amazon_processed.csv", embeddings_cache="embeddings_cache.pkl"):
        """Initialize all components."""
        self.load_models()
        self.load_data(csv_path)
        self.load_or_compute_embeddings(embeddings_cache)
