import torch
import torch.nn as nn
import torch.nn.functional as F


class One_Hot_CrossEntropy(nn.Module):
    def __init__(self):
        super(One_Hot_CrossEntropy, self).__init__()

#わからない
    def forward(self, outputs, targets):
        log_x = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(targets * log_x)
        return loss
