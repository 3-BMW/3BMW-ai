import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import timm

class ImageModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ImageModel, self).__init__()

        self.num_classes= num_classes
        self.model= timm.create_model(model_name, num_classes= self.num_classes, pretrained= True)
    
    def forward(self, data):
        output= self.model(data)

        return output