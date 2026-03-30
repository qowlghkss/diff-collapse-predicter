import torch
from torchvision import transforms
from PIL import Image
import itertools
import numpy as np
import torch.nn.functional as F

class MultiViewConsistencyEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Load DINOv2 model for feature extraction
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.model.eval()
        
        # DINOv2 transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def split_views(self, combined_img: Image.Image) -> list[Image.Image]:
        """Splits a horizontally concatenated 4-view image into separate images."""
        w, h = combined_img.size
        view_w = w // 4
        views = []
        for i in range(4):
            # left, upper, right, lower
            box = (i * view_w, 0, (i + 1) * view_w, h)
            views.append(combined_img.crop(box))
        return views

    @torch.no_grad()
    def compute_consistency(self, img: Image.Image) -> tuple[float, float]:
        """
        Computes pairwise cosine similarity between 4 views in a concatenated image.
        Returns:
            mean_similarity: float
            variance_similarity: float
        """
        views = self.split_views(img)
        if len(views) != 4:
            raise ValueError(f"Expected 4 views, but got {len(views)}")
            
        tensors = []
        for v in views:
            t = self.transform(v.convert("RGB")).unsqueeze(0).to(self.device)
            tensors.append(t)
            
        # Batch items for faster feature extraction
        batch = torch.cat(tensors, dim=0)
        
        # Extract features (CLS token)
        features = self.model(batch) # [4, feature_dim]
        # Normalize features for cosine similarity
        features = F.normalize(features, p=2, dim=-1)
        
        pairwise_sims = []
        # Calculate pairwise cosine similarity (6 pairs for 4 views)
        for i, j in itertools.combinations(range(4), 2):
            sim = (features[i] * features[j]).sum().item()
            pairwise_sims.append(sim)
            
        mean_sim = float(np.mean(pairwise_sims))
        var_sim = float(np.var(pairwise_sims))
        
        return mean_sim, var_sim

    @torch.no_grad()
    def compute_all_metrics(self, img: Image.Image) -> tuple[float, float, float, np.ndarray]:
        """
        Computes pairwise cosine similarity between 4 views in a concatenated image.
        Returns:
            mean_similarity: float
            variance_similarity: float
            min_similarity: float
            sim_matrix: np.ndarray (4x4)
        """
        views = self.split_views(img)
        if len(views) != 4:
            raise ValueError(f"Expected 4 views, but got {len(views)}")
            
        tensors = []
        for v in views:
            t = self.transform(v.convert("RGB")).unsqueeze(0).to(self.device)
            tensors.append(t)
            
        # Batch items for faster feature extraction
        batch = torch.cat(tensors, dim=0)
        
        # Extract features (CLS token)
        features = self.model(batch) # [4, feature_dim]
        # Normalize features for cosine similarity
        features = F.normalize(features, p=2, dim=-1)
        
        # Full pairwise similarity matrix
        sim_matrix = (features @ features.T).cpu().numpy()
        
        pairwise_sims = []
        # Calculate pairwise cosine similarity (6 pairs for 4 views)
        for i, j in itertools.combinations(range(4), 2):
            pairwise_sims.append(sim_matrix[i, j])
            
        mean_sim = float(np.mean(pairwise_sims))
        var_sim = float(np.var(pairwise_sims))
        min_sim = float(np.min(pairwise_sims))
        
        return mean_sim, var_sim, min_sim, sim_matrix
