import torch
import torch.nn as nn

import numpy as np

class SelfAttention(nn.Module):
  def __init__(self,in_channels,out_channels,heads = 8,dropout = 0.1):
    super().__init__()
    assert out_channels % heads == 0, 'wrong channels or heads'
    self.dk = out_channels // heads
    self.heads = heads
    self.out_channels = out_channels
    self.k = nn.Linear(in_features=in_channels,out_features=out_channels)
    self.q = nn.Linear(in_features=in_channels,out_features=out_channels)
    self.v = nn.Linear(in_features=in_channels,out_features=out_channels)

    self.proj = nn.Linear(out_channels,in_channels)
    self.ln = nn.LayerNorm(in_channels)

    self.do = nn.Dropout(dropout)
    
    
  def forward(self,x):
    # x - [b_s,hw,c] -> [b_s,hw,out_ch] -> [b_s,heads,hw,out_ch]
    bs,hw,_ = x.shape
    
    k = self.k(x).view(bs,hw,self.heads,self.dk)
    q = self.q(x).view(bs,hw,self.heads,self.dk)
    v = self.v(x).view(bs,hw,self.heads,self.dk)

    attention = torch.softmax(q@k.mT/self.dk**(0.5),dim = -1)@v
    # [bs,heads,hw,dk] -> [bs,hw,out_ch]
    attention = attention.view(bs,hw,self.out_channels)

    attention = self.do(self.proj(attention))

    return attention
  

class FeedForwardBlock(nn.Module):
  def __init__(self,in_channels,middle_channels):
    super().__init__()
    self.ff = nn.Sequential(
      nn.Linear(in_features=in_channels,out_features=middle_channels),
      nn.LayerNorm(normalized_shape=middle_channels),
      nn.SiLU(),
      nn.Linear(in_features=middle_channels,out_features=in_channels),
      nn.LayerNorm(normalized_shape=in_channels),
      nn.SiLU(),
    )
    self.ln = nn.LayerNorm(in_channels)

  def forward(self,x):

    x_ = self.ff(x)
    x_ = x_ + self.ln(x)
    return x_
  


class TransofrmerEncoder(nn.Module):
  def __init__(self,in_channels,middle_channels,out_channels,dropout = 0.1,heads = 8):
    super().__init__()
    self.self_att = SelfAttention(in_channels=in_channels,out_channels=out_channels,dropout=dropout,heads=heads)
    self.ff = FeedForwardBlock(in_channels=in_channels,middle_channels=middle_channels)
    self.ln1 = nn.LayerNorm(in_channels)
    self.ln2 = nn.LayerNorm(in_channels)

  def forward(self,x):
    x = self.self_att(x) + self.ln1(x)
    x = self.ff(x) + self.ln2(x)

    return x
  

class Transformer(nn.Module):
  def __init__(self,in_channels,middle_channels,out_channels,num_classes = 10,N_blocks=1,dropout = 0.1,heads = 8):
    super().__init__()
    self.blocks = nn.ModuleList([TransofrmerEncoder(in_channels,middle_channels,out_channels,dropout=dropout,heads = heads) for i in range(N_blocks)])
    

  def forward(self,x):
    for block in self.blocks:
      x = block(x)
    
    return x

