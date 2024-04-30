import torch
import torch.nn as nn

import numpy as np

from transofrmer_utils import Transformer

class VisionTransformerClassifier(nn.Module):
  def __init__(self,in_channels,middle_channels,embedded_channels,out_channels,patch_size,N_blocks=1,num_classes=10,dropout=0.1,heads = 8):
    super().__init__()
    
    self.patcher = nn.Sequential(
      nn.Conv2d(in_channels=in_channels,out_channels=embedded_channels,kernel_size=patch_size,stride=patch_size,padding=0),
      nn.BatchNorm2d(embedded_channels),
      nn.Flatten(start_dim=2)
    )
    #x - [bs,e,ps^2]
    self.extra_emb = nn.Embedding(1,embedding_dim=embedded_channels)
    self.pos = nn.Embedding(patch_size**2 + 1,embedding_dim=embedded_channels)

    self.transformer = Transformer(
      in_channels=embedded_channels,
      middle_channels=middle_channels,
      out_channels=out_channels,
      num_classes=num_classes,
      N_blocks=N_blocks,
      dropout=dropout,
      heads=heads
    )

    self.mlp = nn.Sequential(
      nn.Linear(embedded_channels,middle_channels),
      nn.LayerNorm(middle_channels),
      nn.SiLU(),
      nn.Linear(middle_channels,num_classes),
      nn.LayerNorm(num_classes),
    )

  def forward(self,x):
    x = self.patcher(x)
    bs,e,ps = x.shape

    extra_emb = self.extra_emb(torch.zeros(size=(1,),dtype=torch.int64,device=x.device)).expand(bs,-1,-1)

    x = x.view(bs,ps,e)

    x = torch.cat((x,extra_emb),dim = 1)
    
    pos = self.pos(torch.arange(0,ps + 1,dtype=torch.int64,device=x.device))

    x += pos

    x = self.transformer(x)
    
    x = self.mlp(x[:,0])

    return torch.softmax(x,dim = -1)
  



