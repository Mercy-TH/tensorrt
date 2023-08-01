import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dataloader import Datasets
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
mode_path = '../model/model.pth'
model_name = 'resnext101_32x8d'
train_data_dir = '../data/dogs-vs-cats_small/train'
dataset = Datasets(train_data_dir)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
model = Model()
if os.path.exists(mode_path):
    model.load_state_dict(torch.load(mode_path, map_location=device))
    print("Successful load weight.")
else:
    print("Not successful load weight.")


lr_fc = 0.01
lr_backbone = lr_fc * 0.01
params = [
    {'params': model.backbone.parameters(), 'lr': lr_backbone},
    {'params': model.liner.parameters(), 'lr': lr_fc},
]

optimizer = optim.Adam(params, lr=0.01)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
loss_func = nn.CrossEntropyLoss()
model = model.to(device)
model.train()
epochs = 100
error = 10000

for e in range(1, epochs+1):

    for i, (x, y) in enumerate(tqdm(dataloader)):
    # for (x, y) in tqdm(dataloader):
        if x is None or y is None:
            continue
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < error:
            torch.save(model.state_dict(), '../model/best_model.pth')
            error = loss.item()
            print(f' epoch ===>> {e} ========= train_loss ====>> {loss.item()} ===== error ======>> {error} ====')
        print(f' epoch ===>> {e} ==== train_loss ====>> {loss.item()} ===== error ======>> {error} ====')
        torch.save(model.state_dict(), mode_path)


