import torch.nn as nn


class DiscriminatorShare(nn.Module):
    """
        构建判别器：（共享生成器中的编码器）

        参数:
        * hidden_dim
    """


class DiscriminatorNoShare(nn.Module):
    """
        构建判别器：（独立编码）

        参数:
        * hidden_dim 隐藏层维度
    """
