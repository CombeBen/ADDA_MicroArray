import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from ADDA_Perso.module import GradientReversal

class Extractor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc1_bn = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2_bn = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256,128)
        self.fc3_bn = torch.nn.BatchNorm1d(128)
        self.apply(self._init_weights)
      
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        return x
    
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128,1)
        self.fc2 = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(128,32)
        self.l2 = torch.nn.Linear(32,8)
        self.l3 = torch.nn.Linear(8,1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    
class Wass_Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(128,32)
        self.l2 = torch.nn.Linear(32,1)
        #self.l3 = torch.nn.Linear(8,1)
        self.apply(self._init_weights)
      
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        #x = F.relu(self.l3(x))
        return x

def compute_gradient_penalty(discriminator, extractor_source, extractor_target, real, fake, device):

    alpha = torch.tensor(np.random.random((real.size(0),1))).to(device)
    
    interpolates = (alpha*real+((1-alpha)*fake)).requires_grad_(True).to(torch.float32)
    
    d_interpolates = discriminator(extractor_source(interpolates))
    
    fake_out = torch.ones([real.size(0),1], dtype=torch.float32, device=device).requires_grad_(False)

    gradients = autograd.grad(
        outputs = d_interpolates,
        inputs = interpolates,
        grad_outputs = fake_out,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(len(gradients),-1)
    gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
    return gradient_penalty

#Méthode permettant de geler les poids d'un réseau
def model_freeze(model):
    for param in model.parameters():
        param.requires_grad = False

#Méthode permettant de dégeler les poids d'un réseau
def model_unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

#Objet Log néccessaire dans l'utilisation de la recherche d'une baseline pour la distinction de domaine
class LogBaseline:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.df_results = pd.DataFrame(columns=["id","epoch","loss","accuracy_train","accuracy_test","accuracy_val","gradient_pen"])
    
    #Enregistre les statistiques d'une époque d'entrainement
    def log_epoch(self, epoch=None, loss=None, accuracy_train=None, accuracy_test=None, accuracy_val=None, gradient_pen = None):
        new_serie = pd.Series([f"run-{self.counter}",epoch, loss, accuracy_train, accuracy_test, accuracy_val,gradient_pen],index= self.df_results.columns,)
        self.df_results = self.df_results.append(new_serie,ignore_index=True)
        
    #Assure l'avancement des logs internes
    def next_run(self):
        self.counter+=1
    
    #Sauvegarde le log au format CSV
    def save_csv(self):
        self.df_results.to_csv(f"{self.name}.csv")

class LogRes:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.df_results = pd.DataFrame(columns=["id","epoch","d_loss","d_loss_test","d_loss_val","extract_loss","d_test_acc","d_val_acc","classifier_acc_test_before","classifier_acc_val_before","classifier_acc_test_after","classifier_acc_val_after","gradient_pen"])
    
    def log_epoch(self, epoch=None, d_loss=None, d_loss_test=None, d_loss_val=None, extract_loss=None, d_test_acc=None, d_val_acc=None, classifier_acc_test_before=None, classifier_acc_val_before=None, classifier_acc_test_after=None, classifier_acc_val_after=None,gradient_penalty=None):
        new_serie = pd.Series([f"run-{self.counter}",epoch,d_loss, d_loss_test, d_loss_val,extract_loss, d_test_acc, d_val_acc, classifier_acc_test_before, classifier_acc_val_before, classifier_acc_test_after, classifier_acc_val_after,gradient_penalty], index=self.df_results.columns,)
        self.df_results = self.df_results.append(new_serie, ignore_index=True)
    
    def next_run(self):
        self.counter += 1
    
    def save_csv(self):
        self.df_results.to_csv(f"{self.name}.csv")

class LogResWass:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.df_results = pd.DataFrame(columns=["id","epoch","total_loss","source_loss","target_loss","diff","gradient_pen", "class_acc_val","class_acc_test"])
    
    def log_epoch(self, epoch=None,total_loss=None, source_loss = None, target_loss=None, diff=None,gradient_pen=None, class_acc_val=None, class_acc_test = None):
        new_serie = pd.Series([f"run-{self.counter}", epoch, total_loss, source_loss, target_loss, diff, gradient_pen, class_acc_val, class_acc_test], index=self.df_results.columns,)
        self.df_results = self.df_results.append(new_serie, ignore_index=True)
    
    def next_run(self):
        self.counter += 1
    
    def save_csv(self):
        self.df_results.to_csv(f"{self.name}.csv")
        