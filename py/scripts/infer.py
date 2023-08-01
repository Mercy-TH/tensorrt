import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dataloader import Datasets
from model import Model
from PIL import Image
from torchvision import transforms
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
mode_path = '../model/best_model.pth'
model_name = 'resnext101_32x8d'
model = Model()
if os.path.exists(mode_path):
    model.load_state_dict(torch.load(mode_path, map_location=device))
    print("Successful load weight.")
else:
    print("Not successful load weight.")

model = model.to(device)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

label_dic = {0: 'cat', 1: 'dog'}

# img_path = '../data/dogs-vs-cats_small/test/cats/cat.1500.jpg'
img_path = '../data/dogs-vs-cats_small/test/dogs/dog.1500.jpg'
img = Tensor(preprocess(Image.open(img_path))).unsqueeze(0).to(device)
pred = model(img)
print(pred)
pred_label = label_dic[torch.argmax(pred).item()]
print(pred_label)
