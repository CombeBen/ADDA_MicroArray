import torch
import pandas as pd
import torch.nn.functional as F
from ADDA_Perso.module import GradientReversal

class ExtractorShifted(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc1_bn = torch.nn.BatchNorm1d(512)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2_bn = torch.nn.BatchNorm1d(256)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.fc3 = torch.nn.Linear(256,128)
        self.fc3_bn = torch.nn.BatchNorm1d(128)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.fc4 = torch.nn.Linear(128,32)
        self.fc4_bn = torch.nn.BatchNorm1d(32)
        self.dropout4 = torch.nn.Dropout(0.1)
        self.apply(self._init_weights)
      
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.fc4_bn(self.fc4(x)))
        return x
    
class ExtractorGRL(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,1024)
        self.fc1_bn = torch.nn.BatchNorm1d(1024)
        self.fc2 = torch.nn.Linear(1024,512)
        self.fc2_bn = torch.nn.BatchNorm1d(512)
        self.fc3 = torch.nn.Linear(512,256)
        self.fc3_bn = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256,128)
        self.fc4_bn = torch.nn.BatchNorm1d(128)
        self.grl = GradientReversal(alpha=1.)
        self.apply(self._init_weights)
      
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.relu(self.fc4_bn(self.fc4(x)))
        x = self.grl(x)
        return x
    
class ClassifierShifted(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l2 = torch.nn.Linear(32,8)
        self.l3 = torch.nn.Linear(8,1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, x):

        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    
class DiscriminatorShifted(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.l2 = torch.nn.Linear(32,8)
        self.l3 = torch.nn.Linear(8,1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, x):

        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x