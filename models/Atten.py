import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math



# batch_first=True：表示输入 (batch_size, seq_len, feature_dim)  模型学习的是 序列长度（seq_len） 之间的依赖关系

class Attention(nn.Module):
    def __init__(self, input_dim, dim, heads=4, dropout=0.):
        super().__init__()

        dim_head = dim // heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(input_dim, dim, bias=False)
        self.to_k = nn.Linear(input_dim, dim, bias=False)
        self.to_v = nn.Linear(input_dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, source):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(source), self.to_v(source)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = F.softmax(scores, dim=-1, dtype=scores.dtype)  # torch.Size([16, 4, 264, 264])

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # torch.Size([16, 264, 16])

        return self.to_out(out),attn

class PreNormAtt(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2):
        return self.fn(self.norm1(x1), self.norm2(x2))



class SEBlock(nn.Module):    
    def __init__(self, mode, channels, ratio):        
        super(SEBlock, self).__init__() #Sequeeze：通过最大池化或全局平均池化获得1x1xc的特征图        
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)        
        self.max_pooling = nn.AdaptiveMaxPool2d(1)        
        if mode == "max":            
            self.global_pooling = self.max_pooling        
        elif mode == "avg":            
            self.global_pooling = self.avg_pooling 
        #Excitation：经过全连接层，ReLU,全连接层，sigmoid        
        self.fc_layers = nn.Sequential(            
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),            
            nn.ReLU(),            
            nn.Linear(in_features = channels // ratio, out_features = channels , bias = False),        
            )
        self.sigmoid=nn.Sigmoid()

        
    def forward(self, x):        
        b, c, _, _ = x.shape        
        v = self.global_pooling(x).view(b, c)        
        v = self.fc_layers(v).view(b, c, 1, 1)        
        v = self.sigmoid(v)       #  torch.Size([16, 8, 1, 1]) 
        #权重相乘返回        
        return x * v,v



class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpatialAttentionLayer, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", bias=True)
        self.act =  nn.ReLU(inplace=True)
        self.att = nn.Sigmoid()

    def forward(self, x, rx):
        xrx = torch.cat((x, rx), 1)
        xatt = self.att(self.conv2(self.act(self.conv1(xrx))))
        return xatt

class SpattialAttentionBlock(nn.Module):  # att1(x_l[0], x_l_aug)
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpattialAttentionBlock, self).__init__()
        ic = in_channels
        oc = out_channels
        ks = kernel_size
        self.attfeat = SpatialAttentionLayer(ic, oc, ks)
    def forward(self, x, rx):
        f = self.attfeat(x, rx)
        af = x * f
        return af
    
   
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        # self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout = dropout

    def forward(self, blocks, in_feat): # feature(871*260), juzhen(871*871) 
        h = F.relu(self.conv1(blocks, in_feat))
        h = F.dropout(h, self.dropout)
        # h = F.relu(self.conv2(h, in_feat))
        return h   
    
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features  # 输入特征的维度（每个节点的特征数）。
        self.out_features = out_features  # 输出特征的维度（每个节点的输出特征数）。
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # 权重矩阵
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Xavier初始化
    
    def forward(self, input, adj):
        """
        input: (B, N_max, in_features)  # B是batch size，N_max是batch中最大节点数，in_features是每个节点的特征数
        adj: (B, N_max, N_max)  # 邻接矩阵，表示每个图的节点之间的连接关系
        """
        
        # 计算每个图的节点特征矩阵乘以权重矩阵
        support = torch.matmul(input, self.W)  # support: (B, N_max, out_features)
        
        # 使用批量矩阵乘法，将邻接矩阵与支持矩阵相乘
        output = torch.bmm(adj, support)  # output: (B, N_max, out_features)
        
        return output

class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 初始化参数
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.attention_w = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.attention_w.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: (B, N, in_features)  # 批次大小 B，每个图 N 个节点，每个节点 in_features 维特征
        adj: (B, N, N)  # 批次大小 B，每个图的邻接矩阵，形状为 (N, N)
        """
        
        # 批量图卷积：每个图的节点特征矩阵与权重矩阵相乘
        hidden = torch.matmul(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)

        # 准备注意力输入，得到每对节点特征的组合
        attention_input = self._prepare_attentional_mechanism_input(hidden)
        
        # 计算注意力得分
        e = self.leakyrelu(torch.matmul(attention_input, self.attention_w).squeeze(2))  # e.shape: (B, N, N)

        # 使用邻接矩阵来屏蔽不相连的节点对的注意力
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e * adj, zero_vec)  # e.shape: (B, N, N), adj.shape: (B, N, N)
        
        # Softmax归一化
        attention = F.softmax(attention, dim=2)  # 注意力归一化，按每行归一化

        # Dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 使用注意力对隐藏状态进行加权求和
        h_prime = torch.matmul(attention, hidden)  # h_prime.shape: (B, N, out_features)

        if self.concat:
            return F.elu(h_prime)  # 使用ELU激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        准备注意力输入，将节点特征拼接为每对节点的特征组合
        Wh: (B, N, out_features)
        """
        B, N, _ = Wh.size()  # B是批次大小，N是节点数

        # 将节点特征重复，以便形成每对节点特征组合
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # (B, N*N, out_features)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # (B, N, out_features)

        # 拼接两个重复的节点特征
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        # 将形状从 (B, N*N, 2*out_features) 转换为 (B, N, N, 2*out_features)
        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)

