import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
from torch import einsum
from .D_model import Embedding
from .T_N_model import DT_Block,DN_Block,DD_Block
from .Atten import Attention,PreNormAtt,SpattialAttentionBlock
# from .gr import GR
# from .fusion1 import CPCA,CPCA_ChannelAttention

# B：batch size
# N：多变量序列的变量数
# T：过去序列的长度

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

        x = x * v   # b d n l       
        #权重相乘返回        
        return x,v

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.enc_in =configs.enc_in
        
        self.D=configs.D
        self.K1=configs.K1
        self.S=configs.S
        self.K2=configs.K2
        self.K3=configs.K3
        self.K4=configs.K4

        self.d_model = configs.d_model
        self.head = configs.head

        self.num_layers = configs.num_layers
        

        self.L = self.seq_len // self.S


        self.embed_layer = Embedding(self.K1, self.S, self.D)

        
        self.Time_net = DT_Block(self.K2,self.enc_in*self.D,self.D)
        self.ROI_net = DN_Block(self.K3,self.L*self.D,self.D)

        
        
        self.fusion_layer = PreNormAtt(self.enc_in,Attention(self.enc_in,self.d_model,heads=self.head))

        # self.backbone = nn.ModuleList([StateBlock(self.L, self.enc_in,self.D, self.K4) for _ in range(self.num_layers)])
        
        self.se= SEBlock("max",self.D,1)


       
        self.projection =   nn.Sequential(
                    nn.Linear(self.D*self.enc_in*self.L, 2048),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(2048),
                    nn.Linear(2048, self.D),
                    )      
        self.predictor = nn.Linear(self.D+self.D, 1)


       
    def forward(self, x_enc): # x: [B, N, T]
             
        # 引入D. 将每个变量的时间序列转换为一个高维特征表示。 操作: 使用一维卷积提取时间序列的局部特征  通过卷积核在时间轴上滑动来实现
        x = self.embed_layer(x_enc)  # [B, N, T] -> [B, N, D, L]
  
        B,N,S,L = x.shape

        x_t = self.Time_net(x)
        x_n = self.ROI_net(x) 

        x_t = rearrange(x_t, 'b n s l-> (b s) l n') 
        x_n = rearrange(x_n, 'b n s l-> (b s) l n') 
        x_tn = self.fusion_layer(x_n,x_t)[0]+x_n
        x_tn = rearrange(x_tn, '(b s) l n-> b s n l',b=B) 

        x,w = self.se(x_tn) 
        x =x+x_tn

        # Flatten
        x = rearrange(x, 'b s n l-> b (n s l)') 
        x = self.projection(x)  # [B, N*D] -> [B, 1]

        w =w.squeeze(2).squeeze(2)
        x = torch.cat((x,w),dim=-1)

        x = self.predictor(x)
  
        return x
