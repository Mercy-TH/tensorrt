from torch import nn
from scripts.backbone import Backbone
from torchvision.models import resnet101, resnext101_32x8d


class Model(nn.Module):
    def __init__(self, model_name='resnext101_32x8d'):
        super(Model, self).__init__()
        self.backbone = Backbone(model_name).load_backbone()
        self.liner = nn.Linear(2048, 2, bias=True)

    def forward(self, x):
        x = self.backbone(x).squeeze()
        x = self.liner(x)
        return x
