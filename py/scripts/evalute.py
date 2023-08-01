import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import Datasets
from model import Model

train_data_dir = '../data/dogs-vs-cats_small/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
mode_path = '../model/best_model.pth'
model_name = 'resnext101_32x8d'
dataset = Datasets(train_data_dir)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
model = Model().to(device)
if os.path.exists(mode_path):
    model.load_state_dict(torch.load(mode_path, map_location=device))
    print("Successful load weight.")
else:
    print("Not successful load weight.")
model.eval()

TP, TN, FN, FP = 0, 0, 0, 0
with torch.no_grad():
    for i, (x, y) in enumerate(tqdm(dataloader)):
        if x is None or y is None:
            continue
        x = x.to(device)
        y = y.to(device).long()
        y_hat = model(x)
        A = torch.argmax(y, dim=1)
        B = torch.argmax(y_hat, dim=1)

        TP = torch.sum((A == 1) & (B == 1)).item()
        FP = torch.sum((A == 0) & (B == 1)).item()
        TN = torch.sum((A == 0) & (B == 0)).item()
        FN = torch.sum((A == 1) & (B == 0)).item()

err = 1e-10
# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN + err)
# Calculate Recall (Sensitivity)
recall = TP / (TP + FN + err)
# Calculate Precision
precision = TP / (TP + FP + err)
# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall + err)
print('accuracy:', accuracy)
print('recall:', recall)
print('precision:', precision)
print('f1_score:', f1_score)



