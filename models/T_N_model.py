import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

# B：batch size
# N：多变量序列的变量数
# T：过去序列的长度

# class T_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
#     def __init__(self,K2,N):
#         super(T_Block, self).__init__()
        
#         self.conv = nn.Conv1d(
#             in_channels=N, 
#             out_channels=N, 
#             kernel_size=K2,
#             padding='same'
#             ) 
       
#     def forward(self, x_emb):
#         # x_emb: [B, N, D, L]
#         D = x_emb.shape[-2]
#         x = rearrange(x_emb, 'b n d l -> (b d) n l')          # [B, N, D, L] -> [B, N*D, L]
        
#         x = F.gelu(self.conv(x))

#         x = rearrange(x, '(b d) n l -> b n d l',d=D)                         
        
#         return x 

# class N_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
#     def __init__(self,K3,L):
#         super(N_Block, self).__init__()

#         self.conv = nn.Conv1d(
#             in_channels=L, 
#             out_channels=L, 
#             kernel_size=K3,
#             padding='same'
#             ) 

#     def forward(self, x_emb):
#         # x_emb: [B, N, D, L]
#         D = x_emb.shape[-2]
#         x = rearrange(x_emb, 'b n d l -> (b d) l n')    
        
#         x = F.gelu(self.conv(x))

#         x = rearrange(x, '(b d) l n -> b n d l',d=D)                         
        
#         return x 


# self.bn = nn.BatchNorm1d(in_c)
# x = self.conv(x)
# x =  self.bn(self.conv(x))   
# x =  F.gelu(self.bn(self.conv(x))) 
# x =  self.bn(F.gelu(self.conv(x))) 
# x =  self.conv1(F.gelu(self.conv(x)))  
# x =  F.gelu(self.conv2(F.gelu(self.conv1(x)))) 

class DT_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
    def __init__(self,k,in_c,groups):
        super(DT_Block, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_c,out_channels=in_c,kernel_size=k,padding='same',groups=groups)
    
    def forward(self, x_emb):
        # x_emb: [B, N, D, L]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b n d l -> b (d n) l')          # [B, N, D, L] -> [B, N*D, L]
        x = F.gelu(self.conv(x))
        x = rearrange(x, 'b (d n) l -> b n d l',d=D)                         
        return x 

class DN_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
    def __init__(self,k,in_c,groups):
        super(DN_Block, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_c,out_channels=in_c,kernel_size=k,padding='same',groups=groups)

    def forward(self, x_emb):
        # x_emb: [B, N, D, L]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b n d l -> b (d l) n')          # [B, N, D, L] -> [B, N*D, L]
        x = F.gelu(self.conv(x))
        x = rearrange(x, 'b (d l) n -> b n d l',d=D)                         
        return x 
    
class T_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
    def __init__(self,k,in_c):
        super(T_Block, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_c,out_channels=in_c,kernel_size=k,padding='same')

    def forward(self, x):
        # x_emb: [B, N, T]    
        x = F.gelu(self.conv(x)) 
        return x 

class N_Block(nn.Module): 
    def __init__(self,k,in_c):
        super(N_Block, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_c,out_channels=in_c,kernel_size=k,padding='same')
        
    def forward(self, x): 
        x = rearrange(x, 'b n t -> b t n')       
        x = F.gelu(self.conv(x))
        x = rearrange(x, 'b t n -> b n t')  
        return x 
    
class DD_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
    def __init__(self,k,in_c,groups):
        super(DD_Block, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_c,out_channels=in_c,kernel_size=k,padding='same',groups=groups)

    def forward(self, x_emb):
        # x_emb: [B, N, D, L]
        N = x_emb.shape[1]
        x = rearrange(x_emb, 'b n d l -> b (n l) d')          # [B, N, D, L] -> [B, N*D, L]
        x = F.gelu(self.conv(x))
        x = rearrange(x, 'b (n l) d -> b n d l',n=N)                         
        return x 


# class DD_Block(nn.Module): #  传入的参数，序列中变量个数（N）、每个变量的特征维度（D）、卷积核的大小（kernel_size）以及在全连接前馈网络（FFN）中次数（r）。
#     def __init__(self,in_c):
#         super(DD_Block, self).__init__()
        
#         self.conv = nn.Conv1d(in_channels=in_c,out_channels=in_c,kernel_size=1,padding='same')

#     def forward(self, x_emb):
#         # x_emb: [B, N, D, L]
#         N = x_emb.shape[1]
#         x = rearrange(x_emb, 'b n d l -> b d (n l)')          # [B, N, D, L] -> [B, N*D, L]
#         x = F.gelu(self.conv(x))
#         x = rearrange(x, 'b d (n l) -> b n d l',n=N)                         
#         return x 