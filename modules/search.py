from typing import List
import torch

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
    def __init__(self, embeddings: torch.Tensor, keys: List[str], n_results: int):
        super().__init__()
        self.embeddings = embeddings
        self.n_results = n_results
        self.keys = keys
    
    def search(self, query: torch.Tensor):
        indices = k_nearest(query, self.embeddings, n_results=self.n_results)
        keys = [self.keys[i] for i in indices]
        embeddings = self.embeddings[indices]
        return keys, embeddings
        