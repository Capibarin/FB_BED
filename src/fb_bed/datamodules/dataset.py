from typing import Tuple

import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, y, x, theta, eps, dtype=torch.float32):
        self.y = torch.as_tensor(y, dtype=dtype)
        self.x = torch.as_tensor(x, dtype=dtype)
        self.theta = torch.as_tensor(theta, dtype=dtype)
        self.eps = torch.as_tensor(eps, dtype=dtype)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.y[idx], self.x[idx], self.theta[idx], self.eps[idx]
