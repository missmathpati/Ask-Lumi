"""
Script to save the fine-tuned CLIP head from the notebook.
Run this in the same environment where you trained the model.
"""

import torch
from torch import nn
from transformers import CLIPModel
import numpy as np

# This should match your notebook setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

# Load base model to get embed_dim
base_model = CLIPModel.from_pretrained(model_name)
embed_dim = base_model.config.projection_dim  # 512

# Define the head class (must match notebook exactly)
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

print("=" * 60)
print("INSTRUCTIONS:")
print("=" * 60)
print("1. Open your RAG_Model V2.ipynb notebook")
print("2. After training the head (after Cell 34), run this code:")
print()
print("   torch.save(head.state_dict(), 'clip_head.pth')")
print("   print('Head saved to clip_head.pth')")
print()
print("3. This will save the fine-tuned head weights")
print("4. The Streamlit app will automatically load it")
print("=" * 60)

