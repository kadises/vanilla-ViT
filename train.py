import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import Accuracy

from VIT import VisionTransformerClassifier
from utils import get_data,avg,saver


def train(args):
  lr = args['lr']

  model = VisionTransformerClassifier(
    in_channels=args['in_channels'],
    embedded_channels=args['embedded_channels'],
    middle_channels=args['middle_channels'],
    out_channels=args['out_channels'],
    patch_size=args['patch_size'],
    N_blocks=args['N_blocks'],
    num_classes=args['num_classes'],
    dropout=args['dropout'],
    heads=args['heads']
  ).to(args['device'])

  optim = torch.optim.AdamW(model.parameters(),lr = lr)

  train,test = get_data(args['path'],args['batch_size'],args['image_size'])

  criterion = nn.BCELoss()

  print(f'Число параметров: {sum([p.numel() for p in model.parameters()])}')

  if args['load_model']:
    model.load_state_dict(torch.load(args['weights_path']))

  for epoch in range(args['epochs']):
    with tqdm(total=len(train)) as t:
      t.set_description(desc=f'epoch: {epoch + 1}')
      model.train()
      val_loss_avg,train_loss_avg,acc_avg_train,acc_avg_val = [],[],[],[]
      acc = Accuracy(task = 'multiclass',num_classes = 10).to(args['device'])
      for imgs,labels in train:
        optim.zero_grad()
        
        labels = nn.functional.one_hot(labels,num_classes = args['num_classes']).to(args['device']).to(dtype = torch.float32)
        
        imgs = imgs.to(args['device'])

        predict = model.forward(imgs)

        loss = criterion(predict,labels)
        train_loss_avg.append(loss)
        tla = avg(train_loss_avg)

        predicted_labels = torch.argmax(predict,dim=-1)
        target_labels = torch.argmax(labels,dim=-1)

        accuracy = acc(predicted_labels,target_labels)
        acc_avg_train.append(accuracy) 

        accuracy_avg = avg(acc_avg_train)
        loss.backward()
        optim.step()

        t.set_postfix(loss = loss.item(),avg_train_loss=tla.item(),accuracy_train = accuracy_avg.item())
        t.update()

    saver(model,args['weights_path'])
    model.eval()
    with tqdm(total = len(test)) as t:
      with torch.no_grad():
        for imgs,labels in test:
          labels = nn.functional.one_hot(labels,num_classes = 10).to(args['device']).to(dtype = torch.float32)
          imgs = imgs.to(args['device'])

          predict = model.forward(imgs)

          loss = criterion(predict,labels)
          val_loss_avg.append(loss)
          vla = avg(val_loss_avg)

          predicted_labels = torch.argmax(predict,dim=-1)
          target_labels = torch.argmax(labels,dim=-1)

          accuracy = acc(predicted_labels,target_labels)
          acc_avg_val.append(accuracy) 

          accuracy_avg = avg(acc_avg_val)

          t.set_postfix(val_loss = loss.item(),avg_loss_val = vla.item(),accuracy_val = accuracy_avg.item())
          t.update()


args = {'path': '/home/viaznikov/datasets',
        'lr':1e-5,
        'epochs':100,
        'device':'cuda',
        'batch_size':64,
        'image_size':64,
        'in_channels':1,
        'embedded_channels':128,
        'middle_channels':256,
        'out_channels':256,
        'patch_size':8,
        'N_blocks':1,
        'num_classes':10,
        'dropout':0.1,
        'heads':8,
        'weights_path':'vit_0.0.1.pth',
        'load_model':True
        }

train(args)
    

