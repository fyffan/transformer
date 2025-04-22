"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

# 层归一化
# 每一个样本的位置向量单独归一化，与 batch size 无关
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        # gamma 和 beta 是可学习参数
        # gamma 是缩放因子（scale），beta 是偏移因子（shift）
        # 归一化之后再用这两个参数恢复模型的表达能力
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        '''
        对最后一个维度（即每个 token 的向量）做标准化
        mean: 每个位置的均值，var: 方差
        keepdim=True 保持维度一致用于后续广播
        '''
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        # 让模型可以学习到归一化之后每个特征维度的合适“缩放”和“偏移”，从而恢复和增强模型的表达能力。
        return out
