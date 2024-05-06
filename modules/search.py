from typing import List
import torch
from torch.nn.init import orthogonal_

def k_nearest(
        query: torch.Tensor, 
        embeddings: torch.Tensor, 
        n_results: int = 16):
    
    n_items, dim = embeddings.shape
    query = query.view(1, dim)
    dist = torch.cdist(query, embeddings)
    dist = dist.view(n_items)
    indices = torch.argsort(dist)
    return indices[:n_results]


class BruteForceSearch(object):
    def __init__(
            self, 
            embeddings: torch.Tensor, 
            keys: List[str], 
            n_results: int, 
            visualization_dim: int):
        
        super().__init__()
        self.embeddings = embeddings
        self.n_results = n_results
        self.keys = keys
        self.visualization_dim = visualization_dim
        proj = torch.zeros(
            embeddings.shape[-1], visualization_dim, device=embeddings.device)
        orthogonal_(proj)
        self.projection = proj
    
    def visualization(self):
        proj = self.embeddings @ self.projection
        assert proj.shape == (self.embeddings.shape[0], self.visualization_dim)
        return proj
    
    def search(self, query: torch.Tensor):
        indices = k_nearest(query, self.embeddings, n_results=self.n_results)
        keys = [self.keys[i] for i in indices]
        embeddings = self.embeddings[indices]
        return keys, embeddings
        