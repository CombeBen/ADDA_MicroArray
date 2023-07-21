import torch
import pandas as pd
import torch.nn.functional as F
from ADDA_Perso.module import GradientReversal

class Baseline_large(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,1024)
        self.fc1_bn = torch.nn.BatchNorm1d(1024)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(1024,512)
        self.fc2_bn = torch.nn.BatchNorm1d(512)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.fc3 = torch.nn.Linear(512,256)
        self.fc3_bn = torch.nn.BatchNorm1d(256)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.fc4 = torch.nn.Linear(256,128)
        self.fc4_bn = torch.nn.BatchNorm1d(128)
        self.fc5 = torch.nn.Linear(128,64)
        self.fc5_bn = torch.nn.BatchNorm1d(64)
        self.fc6 = torch.nn.Linear(64,1)
        self.fc7 = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight,gain = 1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
                
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.fc4_bn(self.fc4(x)))
        x = F.relu(self.fc5_bn(self.fc5(x)))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

class Baseline(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc1_bn = torch.nn.BatchNorm1d(512)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2_bn = torch.nn.BatchNorm1d(256)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.fc3 = torch.nn.Linear(256,128)
        self.fc3_bn = torch.nn.BatchNorm1d(128)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.l1 = torch.nn.Linear(128,32)
        self.l2 = torch.nn.Linear(32,8)
        self.l3 = torch.nn.Linear(8,1)
        self.fc5 = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight,gain = 1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
                
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.fc5(x)
        return x
    
class Baseline_few(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,128)
        self.fc1_bn = torch.nn.BatchNorm1d(128)
        self.fc4 = torch.nn.Linear(128,1)
        self.fc5 = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight,gain = 1.0)
            if module.bias is not None:
                module.bias.data.zero_()
            
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
                
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x)) 
        return x

class Baseline_middle(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,256)
        self.fc1_bn = torch.nn.BatchNorm1d(256)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(256,128)
        self.fc2_bn = torch.nn.BatchNorm1d(128)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.fc3 = torch.nn.Linear(128,1)
        self.fc4 = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight,gain = 1.0)
            if module.bias is not None:
                module.bias.data.zero_()
            
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
                
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
def forward_hook(list_extracted):
    def hook(model, input, output):
        list_extracted.extend(output.cpu().detach().numpy().tolist())
    return hook

class Baseline_noBatch(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,128)
        self.fc4 = torch.nn.Linear(128,1)
        self.fc5 = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight,gain = 1.0)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x)) 
        return x
    
