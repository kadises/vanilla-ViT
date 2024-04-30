import torch
import numpy as np
from torchvision import transforms,datasets 

def get_data(path,batch_size,image_size):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size,image_size)),
    transforms.Normalize((0.5),(0.5))
  ])

  train_data = datasets.MNIST(path,train=True,transform=transform)
  test_data = datasets.MNIST(path,train=False,transform=transform)

  train = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
  test = torch.utils.data.DataLoader(dataset = test_data,batch_size = batch_size,shuffle=True)

  return train,test


def avg(li,device = 'cuda'):
  return torch.tensor(li).mean(dim=-1).to(device)

def saver(model,weight_path):
  torch.save(model.state_dict(),weight_path)