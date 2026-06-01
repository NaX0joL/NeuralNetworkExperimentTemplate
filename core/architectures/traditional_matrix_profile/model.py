import numpy as np
import stumpy

import torch
from torch import nn, Tensor

from .config import TraditionalMatrixProfileConfig



class TraditionalMatrixProfile(nn.Module):
    def __init__(self, config:TraditionalMatrixProfileConfig):
        super().__init__()
        self.window_size = config.window_size
        
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        return
        
    def forward(self, x:Tensor) -> Tensor:
        batch_size, seq_len = x.shape
        
        device = x.device
        x = x.detach().cpu().numpy()
        
        results = []
        
        for batch_index in range(batch_size):
            mp = compute_1d_matrix_profile(x[batch_index], window_size=self.window_size)
            mp = pad_1d_matrix_profile(mp, target_len=seq_len, pad_side='both')
            results.append(mp)
        
        x = torch.tensor(
            np.stack(results, axis=0), 
            dtype=torch.float32, 
            device=device
        )
        
        x = x + self.dummy_param * 0
        return x



def compute_1d_matrix_profile(time_series:np.ndarray, window_size:int) -> np.ndarray:
    time_series = time_series.astype(np.float64)
    matrix_profile = stumpy.stump(time_series, m=window_size)
    matrix_profile = matrix_profile[:, 0].astype(np.float32)
    return matrix_profile


def pad_1d_matrix_profile(matrix_profile:np.ndarray, target_len:int, pad_side:str) -> np.ndarray:
    pad_width = target_len - matrix_profile.shape[-1]
    pad_value = 0.0
    
    if pad_width <= 0:
        return matrix_profile
    
    if pad_side == 'left':
        left, right = 0, pad_width
    elif pad_side == 'left':
        left, right = pad_width, 0
    else:
        left  = pad_width // 2
        right = pad_width - left
    
    pad_shape = [(0, 0)] * (matrix_profile.ndim - 1) + [(left, right)]
    
    matrix_profile = np.pad(matrix_profile, pad_shape, mode="constant", constant_values=pad_value)
    return matrix_profile