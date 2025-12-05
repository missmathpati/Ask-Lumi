"""
Retrieval module for text and image-based product search.
"""

import torch
import numpy as np
from PIL import Image


class ProductRetriever:
    """Handles product retrieval using CLIP embeddings."""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.base_model = model_loader.base_model
        self.head = model_loader.head
        self.processor = model_loader.processor
        self.device = model_loader.device
        self.df = model_loader.df
        self.index = model_loader.index
        self.item_embs = model_loader.item_embs
        
    def encode_query_text(self, query: str):
        """Encode a text query into an embedding."""
        inputs = self.processor(
            text=[query],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            txt_feat = self.base_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            _, _, _, txt_proj = self.head(
                image_features=torch.zeros_like(txt_feat),
                text_features=txt_feat
            )
        
        q = txt_proj[0].cpu().numpy()
        q = q / np.linalg.norm(q)
        return q.astype("float32")
    
    def encode_query_image(self, image_path_or_pil):
        """Encode an image query into an embedding."""
        if isinstance(image_path_or_pil, str):
            try:
                image = Image.open(image_path_or_pil).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), (255, 255, 255))
        else:
            image = image_path_or_pil.convert("RGB")
        
        inputs = self.processor(
            text=None,
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            img_feat = self.base_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            _, _, img_proj, _ = self.head(
                image_features=img_feat,
                text_features=torch.zeros_like(img_feat)
            )
        
        q = img_proj[0].cpu().numpy()
        q = q / np.linalg.norm(q)
        return q.astype("float32")
    
    def retrieve_products_from_query_vec(self, q_vec, k=5):
        """
        Retrieve top-k products using a query embedding vector.
        
        Args:
            q_vec: (d,) normalized query embedding
            k: number of results to return
            
        Returns:
            DataFrame with top-k products and similarity scores
        """
        sims = self.item_embs @ q_vec
        topk_idx = np.argsort(-sims)[:k]
        results = self.df.iloc[topk_idx].copy()
        results["similarity"] = sims[topk_idx]
        return results
    
    def retrieve_by_text_query(self, user_query, k=5):
        """Retrieve products using a text query."""
        q_vec = self.encode_query_text(user_query)
        return self.retrieve_products_from_query_vec(q_vec, k=k)
    
    def retrieve_by_image_query(self, image_path_or_pil, k=5):
        """Retrieve products using an image query."""
        q_vec = self.encode_query_image(image_path_or_pil)
        return self.retrieve_products_from_query_vec(q_vec, k=k)

