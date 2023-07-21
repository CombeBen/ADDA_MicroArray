import torch as torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time

class Network(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc1_bn = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2_bn = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256,128)
        self.fc3_bn = torch.nn.BatchNorm1d(128)
        self.fc4 = torch.nn.Linear(128,output_dim)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1_bn(self.fc1(x)))
        x = torch.nn.functional.relu(self.fc2_bn(self.fc2(x)))
        x = torch.nn.functional.relu(self.fc3_bn(self.fc3(x)))
        x = torch.nn.functional.relu(self.fc4(x))
        return x
    
def forward_hook(list_extracted):
    def hook(model, input, output):
        list_extracted.append(output.detach())
    return hook

class ExtractorAddA(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc1_bn = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2_bn = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256,128)

        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1_bn(self.fc1(x)))
        x = torch.nn.functional.relu(self.fc2_bn(self.fc2(x)))
        x = torch.nn.functional.relu(self.fc3(x))
        return x

class ClassifierAddA(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Sigmoid(128,1)
        
    def forward(self,x):
        x= self.fc1(x)

def construct_list_hook(list_hooked_data, dataloader_label):
    X_hooked=[]
    X_labels=[]
    for i in range(len(list_hooked_data)):
        for j in range(len(list_hooked_data[i])):
            X_hooked.append(list_hooked_data[i][j].cpu().clone().numpy())
    
    for i in dataloader_label[0]:
        _, y = i
        for j in range(len(y)):
            X_labels.append(y[j].cpu().clone().numpy())
    return X_hooked, X_labels
    
    
    
def transfer_weights(net, weight_path,strict=False):
    net.load_state_dict( torch.load(weight_path),strict=strict)
    
#Méthode utilisée pour entrainer un réseau donné en paramètre pour 1 epoch seulement
def train_epoch(net, dataloader, optimizer, loss_function):
    epoch_loss = 0
    train_steps = 0
    net.train()
    for batch in dataloader :
        X, y = batch
        y = y.unsqueeze(1)
        
        optimizer.zero_grad()
        
        output = net(X.view(-1, X.shape[1]))
        loss = loss_function(output, y)
        epoch_loss += loss.item()
        train_steps +=1
        loss.backward()
        optimizer.step()
    
    return epoch_loss, train_steps


#Méthode utilisée pour Evaluer un réseau donné en paramètre pour 1 epoch seulement
def evaluate_epoch(net, dataloader, loss_function):
    epoch_loss = 0
    eval_steps = 0
    y_true=[]
    y_pred=[]
    activation = torch.nn.Sigmoid()
    net.eval()
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch          #on récupère les valeurs dans le batch, en sachant qu'un batch = [inputs, labels]
            y = y.unsqueeze(1)
            output = net(X.view(-1, X.shape[1]))
            loss = loss_function(output,y)
            epoch_loss+=loss.item()
            eval_steps+=1
            prediction = activation(output).detach().cpu().numpy().round()
            
            y_pred+=prediction.tolist()
            y_true+= y.detach().cpu().numpy().round().tolist()
    
    return epoch_loss, eval_steps, y_pred, y_true
            
def train_and_evaluate_network(config_epoch, dataloaders, net, train = True, test = True, evaluate = True, log = True):
    res=[]
    
    trainset=dataloaders[0]
    testset=dataloaders[1]
    valset=dataloaders[2]
    
    
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    
    
    for epoch in range(config_epoch):
        
        if train:
            train_loss, train_steps = train_epoch(net, trainset, optimizer, loss_function)
            if log:
                print(
                    "Epoch : {:>3}/{} | Train Loss={:.4f}   ||  ".format(
                        epoch+1,
                        config_epoch,
                        train_loss/train_steps,
                    ),
                    end="",
                )
                
        if evaluate:        
            val_loss, val_steps, val_y_pred, val_y_true = evaluate_epoch(net, valset, loss_function)
            if log:
                print(
                    " Val Loss={:.4f} | Val Acc={:.4f}".format(
                        val_loss/val_steps,
                        accuracy_score(val_y_true,val_y_pred)
                    ),
                    end=" ",
                )
                
        if test:        
            test_loss, test_steps, test_y_pred, test_y_true = evaluate_epoch(net, testset, loss_function)
            if log:
                print(
                    "  ||   Test Loss={:.4f} | Test Acc={:.4f}".format(
                        test_loss/test_steps,
                        accuracy_score(test_y_true,test_y_pred)
                    ),
                    end="\n",
                )
    if train:
        res.append(train_loss/train_steps)
    if evaluate:
        res.append(accuracy_score(val_y_true,val_y_pred))
    if test:
        res.append(accuracy_score(test_y_true,test_y_pred))
                
    return res



def train_A_transfer_net_evaluate_B(config_epoch, trainNet, evalNet, trainDataloaders, evalDataloaders):
    start_time = time.process_time()
    train_and_evaluate_network(config_epoch, trainDataloaders, trainNet, log=False)
    print("Pre-training net A done in : % seconds ||  Transfering weigths from net A to net B"%(time.process_time()-start_time))
    evalNet.load_state_dict(trainNet.state_dict())
    return train_and_evaluate_network(config_epoch, evalDataloaders, evalNet)

def train_A_same_net_evaluate_B(config_epoch, trainNet,trainDataloaders, evalDataloaders):
    start_time = time.process_time()
    train_and_evaluate_network(config_epoch, trainDataloaders, trainNet, evaluate=False, test=False, log=False)
    print("Pre-training net done in : % seconds ||  "%(time.process_time()-start_time))
    return train_and_evaluate_network(1, evalDataloaders, trainNet, train=False)
    
def mean_network(config_epoch, nbr_res, net, trainDataloaders, evalDataloaders):
    res=[]
    for i in range(nbr_res):
        print(f"{i}/{nbr_res} réseau")
        res.append(train_A_same_net_evaluate_B( config_epoch, net, trainDataloaders, evalDataloaders))
    return np.array(res).T