from torch import nn
from torchvision.models import resnext101_32x8d


class Backbone:
    def __init__(self, model_name, pretrained=None):
        self.model_name = model_name
        self.pretrained = pretrained

    def load_backbone(self):
        model_names = ['resnext101_32x8d']
        if self.model_name not in model_names: return None
        model = eval(f'{self.model_name}')(weights=self.pretrained)
        backbone = nn.Sequential(*list(model.children())[:-1])
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone

# model_name='resnext101_32x8d'
# bb = Backbone(model_name).load_backbone()