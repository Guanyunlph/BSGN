import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

# B：batch size
# N：多变量序列的变量数
# T：过去序列的长度

class Embedding(nn.Module):
    def __init__(self, K1=7, S=7, D=8):
        super(Embedding, self).__init__()
        self.K1 = K1
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=D, 
            kernel_size=K1, 
            stride=S
            )

    def forward(self, x):  # x: [B, N, T]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, N, T] -> [B, N, 1, T]
        x = rearrange(x, 'b n 1 t -> (b n) 1 t')  # [B, N, 1, T] -> [B*N, 1, T]
        x_pad = F.pad(
            x,
            pad=(0, self.K1-self.S),
            mode='replicate'
            )  # [B*N, 1, T] -> [B*N, 1, T+K1-S]
        
        x_emb = self.conv(x_pad)  # [B*N, 1, T+K1-S] -> [B*N, D, L]
                # 一维卷积操作涉及一个卷积核（或滤波器），它是一个固定大小K1的一维窗口。
                # 这个卷积核在输入序列上滑动，每次移动S个单位（步长），并在每个位置计算卷积核与其覆盖的序列部分的元素乘积之和。 
                # 理解： 移动 L=T/S 次，计算L次，输出L
       
        x_emb = rearrange(x_emb, '(b n) d l -> b n d l', b=B)  # [B*N, D, L] -> [B, N, D, L]

        return x_emb  # x_emb: [B, N, D, L]

