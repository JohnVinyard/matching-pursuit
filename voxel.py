import torch
from torch import nn




class RoomModel(nn.Module):
    
    def __init__(self, room_size: int, voxel_size: int, n_frames: int):
        super().__init__()
        self.room_size = room_size
        self.voxel_size = voxel_size
        self.n_frames = n_frames
        
        self.n_coeffs = voxel_size // 2 + 1
        
        self.mid = room_size // 2
    
        self.responses = nn.Parameter(torch.zeros(1, self.n_coeffs, room_size, room_size).uniform_(1e-12, 0.9))
        self.performance = nn.Parameter(torch.zeros(1, 1, room_size, room_size, n_frames).bernoulli_(p=0.01))
    
    def forward(self) -> torch.Tensor:
        output_frames = []
        
        for i in range(self.n_frames):
            pass