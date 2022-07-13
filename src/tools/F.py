import torch
from torch import Tensor


def get_sparsity(data: Tensor):
    """计算矩阵的稀疏度 1 - data.nonzeros / data.size
    """
    return 1 - torch.norm(data, p=0) / data.flatten().shape[0]