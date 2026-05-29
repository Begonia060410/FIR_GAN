import torch
import torch.nn as nn
class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
    def forward(self,image_pro,pos_0,pos_1):
        loss=torch.mean(torch.square(image_pro[:,0]-pos_0))+torch.mean(torch.square(image_pro[:,1]-pos_1))#image_pro:tensor:2;tensor[1,2]
        return loss
