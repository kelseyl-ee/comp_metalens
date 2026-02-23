import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms


class RetrainFC(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.fc = nn.Linear(D, 10)

    def forward(self, x):
        x = F.relu(x)
        return self.fc(x)
    
    @torch.no_grad()
    def eval_acc(self, loader):
        self.eval()
        device = next(self.parameters()).device
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = self(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / total

