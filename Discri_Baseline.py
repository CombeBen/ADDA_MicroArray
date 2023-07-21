from time import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score
from ADDA_Perso.ADDA import *
from ADDA_Perso.early_stopping import *
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def domain_baseline_train(
    baseline_net, 
    num_epoch, 
    source_dataloaders, 
    target_dataloaders, 
    checkpoint, 
    early_stop = 10, 
    skipping_early_steps = 40,
    weight= 1e-3, 
    l1_decay= 1e-3, 
    l2_decay = 1e-3, 
    device = None, 
    logger = None, 
    weight_dec = False, 
    reg_l1=False, 
    reg_l2=False, 
    early_s = False
):
    if weight_dec :
        optimizer = torch.optim.Adam(baseline_net.parameters(), weight_decay=weight, lr=1e-5)
    else :
        optimizer = torch.optim.Adam(baseline_net.parameters(), weight_decay=0, lr=1e-4)
        
    loss_function = torch.nn.BCELoss()
    
    es = EarlyStopping(patience=early_stop, min_delta=0.0001, log=logger)
    
    for epoch in tqdm(range(num_epoch)):
        
        baseline_net.train()
        n_iters = min(len(source_dataloaders[0]), len(target_dataloaders[0]))
        source_iter, target_iter = iter(source_dataloaders[0]), iter(target_dataloaders[0])
        
        for iter_i in range(n_iters-1):
            source_x, source_y = next(source_iter)
            target_x, target_y = next(target_iter)

            if device :
                source_x = source_x.to(device)
                target_x = target_x.to(device)

            #Output des Inputs de Source et Target depuis leur Extracteur respectif
            source_output = baseline_net(source_x.view(-1,source_x.shape[1]))
            target_output = baseline_net(target_x.view(-1,target_x.shape[1]))
            
            #Création des labels pour Source et Target
            #bs = source_x.size(0)
            source_label = torch.tensor([0]*source_x.size(0), dtype=torch.float).to(device)
            target_label = torch.tensor([1]*target_x.size(0), dtype=torch.float).to(device)
            
            labels = torch.cat([source_label,target_label],dim=0)
            outputs = torch.cat([source_output, target_output],dim=0)

            #Compute de la loss et Update des poids du Discriminator
            loss = loss_function(outputs,labels.unsqueeze(1))
            
            if (reg_l1==True):
                l1_weight = l1_decay
                l1_parameters = []
                for parameter in baseline_net.parameters():
                    l1_parameters.append(parameter.view(-1))
                l1 = l1_weight * baseline_net.compute_l1_loss(torch.cat(l1_parameters))
                loss += l1
            if (reg_l2==True):
                l2_weight = l2_decay
                l2_parameters = []
                for parameter in baseline_net.parameters():
                    l2_parameters.append(parameter.view(-1))
                l2 = l2_weight * baseline_net.compute_l2_loss(torch.cat(l2_parameters))
                loss += l2
    
            loss.backward()
            optimizer.step()
            
            accuracy_train = accuracy_score(outputs.cpu().detach().numpy().round().tolist(),labels.cpu().detach().numpy().tolist())
            
        
        accuracy_test = domain_baseline_eval(baseline_net, source_dataloaders[1], target_dataloaders[1], False, device)
        accuracy_val = domain_baseline_eval(baseline_net, source_dataloaders[2], target_dataloaders[2], False, device)
        
        if early_stop and early_s and epoch > skipping_early_steps:
            es(-accuracy_val)
            if es.early_stop:
                break
        
        if(epoch == (num_epoch-1)):
            _ = domain_baseline_eval(baseline_net, source_dataloaders[0], target_dataloaders[0], True, device)
            accuracy_test = domain_baseline_eval(baseline_net, source_dataloaders[1], target_dataloaders[1], True, device)
            accuracy_val = domain_baseline_eval(baseline_net, source_dataloaders[2], target_dataloaders[2], True, device)
        
        logger.log_epoch(epoch, loss.cpu().detach().numpy(), accuracy_train, accuracy_test, accuracy_val)
        logger.next_run()
        
        if((epoch%checkpoint)==-1 ):
            visualisation(epoch , baseline_net, source_dataloaders, target_dataloaders, device)
    logger.save_csv()
        
def domain_baseline_eval(baseline_net, source_dataloader, target_dataloader, confusion = False, device=None):
    baseline_net.eval()
    n_iters = min(len(source_dataloader), len(target_dataloader))
    source_iter, target_iter = iter(source_dataloader), iter(target_dataloader)
    res = []
    labs = []
    for iter_i in range(n_iters-1):
        source_x, source_y = next(source_iter)
        target_x, target_y = next(target_iter)

        if device :
            source_x = source_x.to(device)
            target_x = target_x.to(device)

        #Output des Inputs de Source et Target depuis leur Extracteur respectif
        source_output = baseline_net(source_x.view(-1,source_x.shape[1]))
        target_output = baseline_net(target_x.view(-1,target_x.shape[1]))
            
        #Création des labels pour Source et Target
        bs = source_x.size(0)
        source_label = torch.tensor([0]*bs, dtype=torch.float).to(device)
        target_label = torch.tensor([1]*bs, dtype=torch.float).to(device)
            
        labels = torch.cat([source_label,target_label],dim=0)
        outputs = torch.cat([source_output, target_output],dim=0)
        
        res.extend(outputs.cpu().detach().numpy().round().tolist())
        labs.extend(labels.cpu().detach().numpy().tolist())
            
    accuracy = accuracy_score(res,labs)
    if(confusion==True):
        cm = confusion_matrix(labs,res,labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = [0,1])
        disp.plot()
        plt.show()
    return accuracy

def visualisation( num_epoch, baseline_net, source_dataloaders, target_dataloaders, device):
    reducer = umap.UMAP()
    pca2D = PCA(n_components=2)
    for loader in range(len(target_dataloaders)):
        X_fc6 = []
        handle = baseline_net.fc4.register_forward_hook(forward_hook(X_fc6))
        joined_outputs, joinded_labels, joined_domain = [], [], []
        source_outputs, source_labels, discri_source_labels = recup_list_donnée( baseline_net, source_dataloaders[loader], device)
        target_outputs, target_labels, discri_target_labels = recup_list_donnée( baseline_net, target_dataloaders[loader], device)
        domain_source = [0]*len(source_labels)
        domain_target = [1]*len(target_labels)
        joined_outputs = source_outputs+target_outputs
        joined_labels = source_labels+target_labels
        joined_domain = domain_source+domain_target
        joined_discri_labels = discri_source_labels + discri_target_labels
        
        print("Epoch number : {:.0f}".format(num_epoch,end="\n"))
        print(print_case_pca(loader))
        
        data_x, data_y = recup(X_fc6)
        
        print(" Domain Ground Truth")
        scatter = plt.scatter(data_x,data_y,s=1,c=joined_domain)
        plt.legend(handles = scatter.legend_elements()[0], labels=['Source','Target'])
        plt.show()
        print(" PCA Discriminator Domain Prediction")
        scatter = plt.scatter(data_x,data_y,s=1,c=joined_discri_labels)
        plt.legend(handles = scatter.legend_elements()[0], labels=['Prédit Source','Prédit Target'])
        plt.show()
        print(" PCA Cancer/Sain Ground Truth")
        scatter = plt.scatter(data_x,data_y,s=1,c=joined_labels)
        plt.legend(handles = scatter.legend_elements()[0], labels=['Cancer','Sain'])
        plt.show()

        
        print("Epoch number : {:.0f}".format(num_epoch,end="\n"))
        print(print_case_umap(loader))
        data_umap = reducer.fit_transform(X_fc6)
        print(" UMAP Domain Ground Truth")
        scatter = plt.scatter(data_umap[:,0],data_umap[:,1],s=1,c=joined_domain)
        plt.legend(handles = scatter.legend_elements()[0], labels=['Source','Target'])
        plt.show()
        print(" UMAP Discriminator Domain Prediction")
        scatter = plt.scatter(data_umap[:,0],data_umap[:,1],s=1,c=joined_discri_labels)
        plt.legend(handles = scatter.legend_elements()[0], labels=['Prédit Source','Prédit Target'])
        plt.show()
        print(" UMAP Cancer/Sain Ground Truth")
        scatter = plt.scatter(data_umap[:,0],data_umap[:,1],s=1,c=joined_labels)
        plt.legend(handles = scatter.legend_elements()[0], labels=['Cancer','Sain'])
        plt.show()
        
        handle.remove()

def recup_list_donnée( baseline, dataloader, device=None):
    outputs, labels, domain_labels = [], [], []
    data_iter = iter(dataloader)
    for k in range(len(dataloader)-1):
        X, Y = next(data_iter)
        if device :
            X = X.to(device)
            Y = Y.to(device)
        output = baseline(X.view(-1, X.shape[1]))
        labels.extend(Y.cpu().detach().numpy().tolist())
        outputs.extend(output.cpu().detach().numpy().tolist())
        domain_labels.extend(output.cpu().detach().numpy().round().tolist())
    return outputs, labels, domain_labels

def print_case_umap(num):
    return{
        0: "Umap Train",
        1: "Umap Test",
        2: "Umap Eval",
    }[num]
        
def print_case_pca(num):
    return{
        0: "PCA Train",
        1: "PCA Test",
        2: "PCA Eval",
    }[num] 

def recup(data):
    x_data, y_data = [],[]
    for p in range(len(data)):
        x, y = (data[p])
        x_data.append(x)
        y_data.append(y)
    return x_data, y_data