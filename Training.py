from time import time
import numpy as np
import torch
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score
from ADDA_Perso.ADDA import *
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
        

def step(net, classifier, X, Y, loss_function):
    '''
    Cette méthode est utilisée pour réaliser le passage de donnée dans un model fournis.
    _________________________________________
    Args :
        net (model) : le model extracteur fournis.
        classifier (model) : le model de décision fournis.
        X (batch) : le Batch des données.
        Y (batch) : le Batch des Labels correspondant aux données.
        loss_function (function) : fonction de coût utilisé
    
    ________________________________________
    Renvoie :
        output (numpy array) : L'output de X passé dans le Net.
        loss (float) : La loss de l'output des données X passé dans le Net en fonction de la loss_function et des labels des données.
    '''
    output = net(X.view(-1, X.shape[1]))
    output = classifier(output)
    loss = loss_function(output, Y)
    return output, loss
    
def train_source_epoch(source_net, classifier, dataloader, optimizer_source, optimizer_classifier, loss_function, device = None):
    '''
    Cette méthode entraine le modèle Classifier et le model Source pour une epoch.
    
    Args :
        Source_net (model) : un modèle de features extractor.
        Classifier (model) : un modèle de classification.
        Dataloader (dataloader) : Le dataloader du domaine Source utilisé.
        optimizer_source (optim) : l'optimizer lié au modèle Source_net.
        optimizer_classifier (optim) : l'optimizer lié au modèle de Classifier.
        loss_function (function) : la fonction de coût utilisée.
    
    Renvoie :
        epoch_loss (float) : la loss pour l'epoch calculé.
        train_steps (int) : le nombre de batch réalisé.
        acc (float) : la précision du Classifier sur les données pour une époch.
    '''
    epoch_loss = 0
    train_steps = 0
    
    source_net.train()
    classifier.train()
    targets, outputs = [], []
    
    for batch in dataloader :
        X, y = batch
        if device :
            X = X.to(device)
            y = y.to(device)
        y = y.unsqueeze(1)
        
        optimizer_source.zero_grad()
        optimizer_classifier.zero_grad()
        
        output, loss = step(source_net, classifier, X, y, loss_function)
        epoch_loss += loss.item()
        train_steps +=1
        
        loss.backward()
        optimizer_source.step()
        optimizer_classifier.step()
        
        targets.extend(y.cpu().detach().numpy().tolist())
        outputs.extend(output.round().tolist())
        torch.cuda.empty_cache()
        
    acc = accuracy_score(targets,outputs)
    return epoch_loss, train_steps, acc

def evaluate_source_epoch(source_net, classifier, dataloader, loss_function, device = None):
    '''
    Evalue l'accuracy et la Loss pour un extracteur et un Classifier donné pour une epoch.
    
    Args :
        Source_net (model) : un modèle de features extractor.
        classifier (model) : un modèle de classification.
        Dataloader (dataloader) : le dataloader contenant les données à évaluer.
        Loss_function (function) : la fonction de coût utilisée.
    
    Renvoie :
        epoch_loss (float) : la loss calculé sur l'Epoch.
        targets (numpy array) : les Labels des données du Dataloader.
        outputs (numpy array) : les Outputs des données du Dataloader passé par le modèle Source et le Classifier.
        acc (float) : l'accuracy sur l'epoch.
    '''
    epoch_loss = 0
    source_net.eval()
    classifier.eval()
    targets, outputs = [], []
    with torch.no_grad():
        for batch in dataloader :
            X, y = batch
            y = y.unsqueeze(1)
            
            if device :
                X = X.to(device)
                y = y.to(device)

            output, loss = step(source_net, classifier, X, y, loss_function)
            epoch_loss += loss.item()
            output = output.cpu().detach().numpy()
            targets.extend(y.cpu().detach().numpy().tolist())
            outputs.extend(output.round().tolist())
            
            torch.cuda.empty_cache()
            
    acc = accuracy_score(targets,outputs)
    
    return epoch_loss, targets, outputs, acc 



def train_source(num_epoch, source_net, classifier_net, dataloaders, loss_function , device,log=False):
    '''
    Entraine le réseau source et le Classifier avec les données du Dataloaders.
    
    Args :
        Num_epoch (int) : Nombre d'epoch réaliser lors de l'entrainement.
        Source_Net (model) : Extractor du domain Source.
        Classifier (model) : Classificateur pour le domain Source.
        Dataloaders (dataloaders) : Objet contenant les 3 dataloader [train; test; eval]
        Loss_Function (function) : Fonction de coût à utiliser pour l'entrainement.
        Log (boolean) = Paramètre pour la verbose de la fonction (Default : False)
    
    '''
    
    optimizer_source = torch.optim.Adam(source_net.parameters(), lr=1e-4)
    optimizer_classifier = torch.optim.Adam(classifier_net.parameters(), lr=1e-5)
    
    for epoch in tqdm(range(num_epoch)):
        train_loss, train_steps, acc = train_source_epoch( source_net, classifier_net, dataloaders[0], optimizer_source, optimizer_classifier, loss_function, device)
        
        if log:
            print(
                "Epoch : {:>3}/{} | Train Loss={:.4f}   || Accuracy : {} ".format(
                    epoch+1,
                    num_epoch,
                    train_loss/train_steps,
                    acc
                ),
                end="\n",
            )

def train_target(
    num_epoch,
    source_net, 
    classifier, 
    target_net, 
    discriminator, 
    target_optimizer,
    d_optimizer,
    loss_function, 
    d_loss_function, 
    source_dataloaders, 
    target_dataloaders,
    target_dataloaders_discri,
    name,
    lambda_gp=1,
    device = None,
    epoch_checkpoint = 1000,
    learn_discrepancy = 5,
    logger = None,
    visualisation = False,
    wasserstein = False,
    log=False):
    
    '''
    Méthode entrainant le réseau de neurones Target avec une méthode adversarial contre le réseau Source.
    
    Args :
        num_epoch (int) : Nombre d'epoch réalisée durant l'entrainement.
        Source_net (model) : Réseau de neurone pré-entrainé du domain Source.
        Classifier (model) : Réseau de neurone pré-entrainé classifiant les résultats du rnn du domain Source.
        Target_net (model) : Réseau de neurone Extracteur du domain Target.
        Discriminateur (model) : Réseau de neurones Classifiant entre le domain Source ou le domain Target.
        Target_Optimizer (optim) : optimisateur des poids du réseau de neurone Extracteur Target.
        D_Optimizer (optim) : optimisateur des poids du réseau de neurones du Discriminateur.
        Loss_function (function) : Fonction de coût pour le réseau de neurone Extracteur Target.
        D_Loss_function (function) : Fonction de coût pour le réseau de neurones Discriminateur.
        Source_Dataloaders (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Source.
        Target_Dataloaders (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Target.
        Target_Dataloaders_Discri (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Target utilisé par le discriminateur.
        name (string) : Chaine de charactère utilisé par les fonctions de visualisation pour enregistrer celles-ci.
        lambda_gp (int) : Coefficient de pénalité utilisé avec la pénalité de gradient dans l'utilisation de la distance de Wasserstein. 
        device (GPU) : Graphic Processor Unit utilisé pour stocker et calculer les opérations d'entrainement des réseaux de neurones. (Default : None)
        epoch_checkpoint (int) : Détermine un nombre d'époch d'entrainement entre chaque visualisation.
        learn_discrepancy (int) : nombre de batch de différence entre l'entrainement du Discriminateur et celui de l'extracteur cible.
        logger (ResLog) : Objet permettant l'enregistrement des différentes métriques de contrôle des réseaux de neurones.
        visualisation (Boolean) : Détermine si les fonctions de Visualisation seront utilisé (celles-ci augmentant la complexité en temps) (Default : False)
        wasserstein (Boolean) : Détermine si le type d'entrainement utilise la distance de Wasserstein. (Default : False)
        log (boolean) : Paramètre de verbose de la méthode. (Default : False)
    '''
    
    if(visualisation==True):
        total_visualisation(0, source_net, target_net, discriminator, source_dataloaders, target_dataloaders, device, name, True)
    
    for i in tqdm(range(num_epoch)):
        if wasserstein : 
            total_loss, target_loss, source_loss, diff, gradient_pen = wasserstein_adversarial_training(
                source_net, 
                target_net, 
                discriminator, 
                target_optimizer, 
                d_optimizer, 
                source_dataloaders, 
                target_dataloaders, 
                target_dataloaders_discri,
                i,
                lambda_gp,
                device,
                log=False
            )
            print(f' Epoch number : {i} Discri total loss: {total_loss}, loss_target: {target_loss}, loss_source: {source_loss}, diff Target-Source {diff}, gradient penalty value : {gradient_pen} \n')
        else:
            d_loss, d_acc, target_loss = adversarial_training(
                source_net, 
                target_net, 
                discriminator, 
                target_optimizer, 
                d_optimizer, 
                loss_function, 
                d_loss_function, 
                source_dataloaders, 
                target_dataloaders, 
                target_dataloaders_discri,
                i,
                device,
                learn_discrepancy,
                log=False
            )
            
        l, t, o, val_acc = evaluate_source_epoch(target_net, classifier, target_dataloaders[2], loss_function, device)
        l, t, o, test_acc = evaluate_source_epoch(target_net, classifier, target_dataloaders[1], loss_function, device)
        
        if (((i==100)or(i==num_epoch-1))and visualisation):
            total_visualisation(i, source_net, target_net, discriminator, source_dataloaders, target_dataloaders, device, name, True)

        logger.log_epoch(i,total_loss.cpu().detach().numpy(), source_loss.cpu().detach().numpy(), target_loss.cpu().detach().numpy(), diff.cpu().detach().numpy(), gradient_pen.cpu().detach().numpy(),val_acc,test_acc)
        logger.next_run()
    logger.save_csv()
    
    direct = "Models/"
    path = os.path.join(direct, name)
    exist = os.path.exists(path)
    if (exist==False):
        os.mkdir(path)
    torch.save(discriminator.state_dict(), path+"/discri")
    torch.save(source_net.state_dict(), path+"/source_net")
    torch.save(target_net.state_dict(), path+"/target_net")
    torch.save(classifier.state_dict(), path+"/classifier")
    
    return
    
    
def adversarial_training(
    source_net, 
    target_net, 
    discriminator, 
    target_optimizer, 
    discriminator_optimizer, 
    target_loss_function, 
    discriminator_loss_function, 
    source_dataloaders, 
    target_dataloaders,
    target_dataloaders_discri,
    num_epoch,
    device = None,
    learn_discrepancy = 5,
    log=False):
    
    '''
    Methode mettant en compétition le réseau Extractor Target et le Discriminator.
    Le Discriminator est entrainé à différencier les éléments venant de l'Extractor Source et Target.
    L'Extractor Target est entrainé à berner le Discriminator en ese faisant passé pour l'Extractor Source.
    
    Args :
        Source_net (model) : Réseau de neurone pré-entrainé du domain Source.
        Target_net (model) : Réseau de neurone Extracteur du domain Target.
        Discriminator (model) : Réseau de neurones Classifiant entre le domain Source ou le domain Target.
        Net_Optimizer (optim) : optimisateur des poids du réseau de neurone Extracteur Target.
        D_Optimizer (optim) : optimisateur des poids du réseau de neurones du Discriminateur.
        Loss_function (function) : Fonction de coût pour le réseau de neurone Extracteur Target.
        D_Loss_function (function) : Fonction de coût pour le réseau de neurones Discriminateur.
        Source_Dataloaders (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Source.
        Target_Dataloaders (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Target.
        Target_Dataloaders_Discri (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Target utilisé par le discriminateur.
        num_epoch (int) : Nombre de l'époch actuellement réalisée.
        device (device) : Appareil sur lequel seront envoyés les données à calculer.
        learn_discrepancy (int) : nombre de batch de différence entre l'entrainement du Discriminateur et celui de l'extracteur cible.
        log (boolean) : Paramètre de verbose de la méthode. (Default : False)
        
    Renvoie :
        d_loss (float) : La loss du Discriminateur.
        net_loss (float) : la loss du Target Extractor.
        d_acc (float) : L'accuracy du Discriminateur.
    '''
    
    #setup des modes pour les modèles.
    #On veut que le Source extractor n'apprenne pas, que le Target Extractor et le Discriminator apprenne. Etape 2 de ADDA.
    source_net.eval()
    for parameter in source_net.parameters():
                parameter.requires_grad = False
    model_freeze(source_net)
    
    #Récupération du nombre minimum de batch entre Target et Source
    n_iters = min(len(source_dataloaders[0]), len(target_dataloaders[0]))
    
    source_iter, target_discriminator_iter, target_extractor_iter = iter(source_dataloaders[0]), iter(target_dataloaders[0]), iter(target_dataloaders_discri[0])
    
    domain_targets, domain_outputs = [], []
    
    for iter_i in range(n_iters-1):
        source_x, source_y = next(source_iter)
        target_x, target_y = next(target_discriminator_iter)
        target_extractor_x, target_discri_y = next(target_extractor_iter)
        
        if device :
            source_x = source_x.to(device)
            target_x = target_x.to(device)
            target_extractor_x = target_extractor_x.to(device)
        
        
        target_net.eval()
        target_net.zero_grad()
        for parameter in target_net.parameters():
                parameter.requires_grad = False
        discriminator.train()
        for parameter in discriminator.parameters():
                parameter.requires_grad = True
        discriminator_optimizer.zero_grad()
        
        #Output des Inputs de Source et Target depuis leur Extracteur respectif
        source_extractor_output = source_net(source_x.view(-1,source_x.shape[1]))
        target_extractor_output = target_net(target_x.view(-1,target_x.shape[1]))
        
        
        #Création des labels pour Source et Target
        bs = source_x.size(0)
        source_discriminator_label = torch.tensor([0]*bs, dtype=torch.float).to(device)
        target_discriminator_label = torch.tensor([1]*bs, dtype=torch.float).to(device)
        
        #Output des outputs des Extracteurs de Source et Target dans le Discriminateur
        source_discriminator_output = discriminator(source_extractor_output)
        target_discriminator_output = discriminator(target_extractor_output)
        
        
        #Concaténation des Outputs du Discriminateur et des labels
        discriminator_label = torch.cat([source_discriminator_label,target_discriminator_label],dim=0)
        discriminator_output = torch.cat([source_discriminator_output, target_discriminator_output],dim=0)

        #Compute de la loss et Update des poids du Discriminator
        discriminator_loss = discriminator_loss_function(discriminator_output,discriminator_label.unsqueeze(1))
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        ##############################
        #Train de l'Extractor
        for parameter in target_net.parameters():
            parameter.requires_grad = True
        target_net.train()
        discriminator.eval()
        target_net.zero_grad()
        for parameter in discriminator.parameters():
            parameter.requires_grad = False
        #Passage des données du Domain Target dans l'Extracteur Target
        d_input_target = target_net(target_extractor_x.view(-1, target_extractor_x.shape[1]))
        #Passage de l'Output de l'Extracteur Target dans le Discriminateur
        d_output_target = discriminator(d_input_target)

        #Calcul de la loss et Minimisation
        target_optimizer.zero_grad()
        target_loss = target_loss_function(d_output_target,source_discriminator_label.unsqueeze(1))
        target_loss.backward()
        target_optimizer.step()
        ##############################
        
            
        domain_outputs.extend(discriminator_output.cpu().detach().numpy().round().tolist())
        domain_targets.extend(discriminator_label.cpu().detach().numpy().tolist())
    d_acc = accuracy_score(domain_outputs,domain_targets)
    return discriminator_loss, d_acc, target_loss

def wasserstein_adversarial_training(
    source_net, 
    target_net, 
    discriminator, 
    target_optimizer, 
    discriminator_optimizer, 
    source_dataloaders, 
    target_dataloaders,
    target_dataloaders_discri,
    num_epoch,
    lambda_gp=10
    device = None,
    log=False):
    
    '''
    Methode mettant en compétition le réseau Extractor Target et le Discriminator.
    Le Discriminator est entrainé à différencier les éléments venant de l'Extractor Source et Target.
    L'Extractor Target est entrainé à berner le Discriminator en ese faisant passé pour l'Extractor Source.
    
    Args :
        Source_net (model) : Réseau de neurone pré-entrainé du domain Source.
        Target_net (model) : Réseau de neurone Extracteur du domain Target.
        Discriminator (model) : Réseau de neurones Classifiant entre le domain Source ou le domain Target.
        Net_Optimizer (optim) : optimisateur des poids du réseau de neurone Extracteur Target.
        D_Optimizer (optim) : optimisateur des poids du réseau de neurones du Discriminateur.
        Source_Dataloaders (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Source.
        Target_Dataloaders (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Target.
        Target_Dataloaders_Discri (list dataloader) : Objet contenant les Dataloaders [train; test; eval] du domain Target utilisé par le discriminateur.
        num_epoch (int) : Nombre de l'époch actuellement réalisée.
        lambda_gp (int) : Coefficient de la pénalité de gradient (Default = 10)
        device (device) : Appareil sur lequel seront envoyés les données à calculer.
        learn_discrepancy (int) : nombre de batch de différence entre l'entrainement du Discriminateur et celui de l'extracteur cible.
        log (boolean) : Paramètre de verbose de la méthode. (Default : False)
        
    Renvoie :
        total_loss (float) : Loss total, c'est-à-dire Target Loss + Source Loss + ( gradient penalty * lambda_gp)
        target_loss (float) : loss de l'extracteur cible
        source_loss (float) : loss de l'extracteur source
        source_loss - target_loss (float) : différence entre la loss source et la loss cible
        (lambda_gp * gradient_penalty) (float)  : pénalité de gradient
    '''
    
    #On veut que le Source extractor n'apprenne pas, que le Target Extractor et le Discriminator apprenne. Etape 2 de ADDA.
    source_net.eval()
    for parameter in source_net.parameters():
                parameter.requires_grad = False
    model_freeze(source_net)
    
    #Récupération du nombre minimum de batch entre Target et Source
    n_iters = min(len(source_dataloaders[0]), len(target_dataloaders[0]))
    
    source_iter, target_discriminator_iter, target_extractor_iter = iter(source_dataloaders[0]), iter(target_dataloaders[0]), iter(target_dataloaders_discri[0])
    
    for iter_i in range(n_iters-1):
        source_x, source_y = next(source_iter)
        target_x, target_y = next(target_discriminator_iter)
        target_extractor_x, target_discri_y = next(target_extractor_iter)
        
        if device :
            source_x = source_x.to(device)
            target_x = target_x.to(device)
            target_extractor_x = target_extractor_x.to(device)
            
            
        #Entrainement du critique
        target_net.eval()
        for parameter in target_net.parameters():
                parameter.requires_grad = False
                
        discriminator.train()
        for parameter in discriminator.parameters():
                parameter.requires_grad = True
        discriminator_optimizer.zero_grad()
        
        #Output des Inputs de Source et Target depuis leur Extracteur respectif
        source_extractor_output = source_net(source_x.view(-1,source_x.shape[1]))
        target_extractor_output = target_net(target_x.view(-1,target_x.shape[1]))
          
        #Output des outputs des Extracteurs de Source et Target dans le Discriminateur
        source_discriminator_output = discriminator(source_extractor_output)
        target_discriminator_output = discriminator(target_extractor_output)
        
        #calcul de la pénalité de gradient
        gradient_penalty = compute_gradient_penalty(discriminator,source_net, target_net, source_x.data, target_x.data, device)

        #Compute de la loss et Update des poids du Discriminator
        total_loss = -torch.mean(source_discriminator_output) + torch.mean(target_discriminator_output) + (lambda_gp * gradient_penalty)
        total_loss.backward()
        discriminator_optimizer.step()
        
        source_loss = torch.mean(source_discriminator_output)
        
        #Entrainement de l'extracteur
        ##############################
        if(iter_i%10==0):
            #Train de l'Extractor
            for parameter in target_net.parameters():
                parameter.requires_grad = True
            target_net.train()
            target_net.zero_grad()

            discriminator.eval()
            for parameter in discriminator.parameters():
                parameter.requires_grad = False

            #Passage des données du Domain Target dans l'Extracteur Target
            d_input_target = target_net(target_extractor_x.view(-1, target_extractor_x.shape[1]))
            #Passage de l'Output de l'Extracteur Target dans le Discriminateur
            d_output_target = discriminator(d_input_target)

            #Calcul de la loss et Minimisation
            target_optimizer.zero_grad()
            target_loss = -torch.mean(d_output_target)
            target_loss.backward()
            target_optimizer.step()
        ##############################

    return total_loss, target_loss, source_loss,source_loss+target_loss, lambda_gp * gradient_penalty


def discriminator_eval_validation(
    source_net, 
    target_net, 
    discriminator, 
    source_dataloaders, 
    target_dataloaders,
    discriminator_loss_function, 
    device, 
    log=False):
    source_net.eval()
    target_net.eval()
    discriminator.eval()
    n_iters = min(len(source_dataloaders[2]), len(target_dataloaders[2]))
    source_iter, target_iter = iter(source_dataloaders[2]), iter(target_dataloaders[2])
    domain_targets, domain_outputs = [], []
    for iter_i in range(n_iters-1):
        source_x, source_y = next(source_iter)
        target_x, target_y = next(target_iter)
        
        if device:
            source_x = source_x.to(device)
            target_x = target_x.to(device)
        
        #Output des Inputs de Source et Target depuis leur Extracteur respectif
        source_extractor_output = source_net(source_x.view(-1,source_x.shape[1]))
        target_extractor_output = target_net(target_x.view(-1,target_x.shape[1]))
        
        
        #Création des labels pour Source et Target
        bs = source_x.size(0)
        source_discriminator_label = torch.tensor([0]*bs, dtype=torch.float).to(device)
        target_discriminator_label = torch.tensor([1]*bs, dtype=torch.float).to(device)
        
        #Output des outputs des Extracteurs de Source et Target dans le Discriminateur
        source_discriminator_output = discriminator(source_extractor_output)
        target_discriminator_output = discriminator(target_extractor_output)
        
        
        #Concaténation des Outputs du Discriminateur et des labels
        discriminator_label = torch.cat([source_discriminator_label,target_discriminator_label],dim=0)
        discriminator_output = torch.cat([source_discriminator_output, target_discriminator_output],dim=0)
        
        d_loss = discriminator_loss_function(discriminator_output,discriminator_label.unsqueeze(1))
        
        domain_outputs.extend(discriminator_output.cpu().detach().numpy().round().tolist())
        domain_targets.extend(discriminator_label.cpu().detach().numpy().tolist())
        
    d_acc = accuracy_score(domain_outputs, domain_targets)
    torch.cuda.empty_cache()
    return d_acc, d_loss

def discriminator_eval_test(
    source_net, 
    target_net, 
    discriminator, 
    source_dataloaders, 
    target_dataloaders, 
    discriminator_loss_function, 
    device=None, 
    log=False):
    source_net.eval()
    target_net.eval()
    discriminator.eval()
    n_iters = min(len(source_dataloaders[1]), len(target_dataloaders[1]))
    source_iter, target_iter = iter(source_dataloaders[1]), iter(target_dataloaders[1])
    domain_targets, domain_outputs = [], []
    for iter_i in range(n_iters-1):
        source_x, source_y = next(source_iter)
        target_x, target_y = next(target_iter)
        
        if device :
            source_x = source_x.to(device)
            target_x = target_x.to(device)
        
        #Output des Inputs de Source et Target depuis leur Extracteur respectif
        source_extractor_output = source_net(source_x.view(-1,source_x.shape[1]))
        target_extractor_output = target_net(target_x.view(-1,target_x.shape[1]))
        
        #Output des outputs des Extracteurs de Source et Target dans le Discriminateur
        source_discriminator_output = discriminator(source_extractor_output)
        target_discriminator_output = discriminator(target_extractor_output)
        
        #Création des labels pour Source et Target
        bs = source_x.size(0)
        source_discriminator_label = torch.tensor([0]*bs, dtype=torch.float).to(device)
        target_discriminator_label = torch.tensor([1]*bs, dtype=torch.float).to(device)
        
        #Concaténation des Outputs du Discriminateur et des labels
        discriminator_label = torch.cat([source_discriminator_label,target_discriminator_label],dim=0)
        discriminator_output = torch.cat([source_discriminator_output, target_discriminator_output],dim=0)
        
        d_loss = discriminator_loss_function(discriminator_output,discriminator_label.unsqueeze(1))
        
        domain_outputs.extend(discriminator_output.cpu().detach().numpy().round().tolist())
        domain_targets.extend(discriminator_label.cpu().detach().numpy().tolist())
        
    
    d_acc = accuracy_score(domain_outputs, domain_targets)
    torch.cuda.empty_cache()
    return d_acc, d_loss



            
def total_visualisation(num_epoch, source_net, target_net, discriminator, source_dataloaders, target_dataloaders, device, name, UMAP_visu = False):
    direct = "Visualisation/ADDA_Visu"
    path = os.path.join(direct, name)
    exist = os.path.exists(path)
    if (exist==False):
        os.mkdir(path)
    path = os.path.join(path,"Epoch_number_{:.0f}".format(num_epoch))
    exist = os.path.exists(path)
    if (exist==False):
        os.mkdir(path)
        
    reducer = umap.UMAP()
    pca2D = PCA(n_components=2)
    ind = 0

    for loader in range(len(target_dataloaders)):
        joined_outputs, joinded_labels, joined_domain = [],[], []
        source_outputs, source_labels = recup_list_donnée(source_net, discriminator, source_dataloaders[loader], device)
        target_outputs, target_labels = recup_list_donnée(target_net, discriminator, target_dataloaders[loader], device)
        domain_source = [0]*len(source_labels)
        domain_target = [1]*len(target_labels)
        joined_outputs = source_outputs+target_outputs
        joined_labels = source_labels+target_labels
        joined_domain = domain_source+domain_target
        
        datas=print_case(ind)
        data_pca = pca2D.fit_transform(joined_outputs)
        ratio = pca2D.explained_variance_ratio_
        ratio=ratio[0]+ratio[1]
        if(ratio>=0.50):
            print(ratio)
            print(" PCA Domain Ground Truth")
            plt.figure(figsize=(12,7))
            scatter = plt.scatter(data_pca[:,0],data_pca[:,1],s=1,c=joined_domain)
            plt.legend(handles = scatter.legend_elements()[0], labels=['Source','Target'])
            plt.savefig(path+"/{:.0f}_PCA_Domain_ground_truth_{setdata}".format(num_epoch,setdata=datas)+".png")
            plt.close()

            print(" PCA Cancer/Sain Ground Truth")
            plt.figure(figsize=(12,7))
            scatter = plt.scatter(data_pca[:,0],data_pca[:,1],s=1,c=joined_labels)
            plt.legend(handles = scatter.legend_elements()[0], labels=['Cancer','Sain'])
            plt.savefig(path+"/{:.0f}_PCA_Cancer_ground_truth_{setdata}".format(num_epoch,setdata=datas)+".png")
            plt.close()
        else :
            print("Variance expliquée trop faible")
        if UMAP_visu : 
            print("Epoch number : {:.0f}".format(num_epoch,end="\n"))
            data_umap = reducer.fit_transform(joined_outputs)
            print(" UMAP Domain Ground Truth")
            plt.figure(figsize=(12,7))
            scatter = plt.scatter(data_umap[:,0],data_umap[:,1],s=1,c=joined_domain)
            plt.legend(handles = scatter.legend_elements()[0], labels=['Source','Target'])
            plt.savefig(path+"/{:.0f}_UMAP_Discriminator_Ground_Truth_{setdata}".format(num_epoch,setdata=datas)+".png")
            plt.close()
            print(" UMAP Cancer/Sain Ground Truth")
            plt.figure(figsize=(12,7))
            scatter = plt.scatter(data_umap[:,0],data_umap[:,1],s=1,c=joined_labels)
            plt.legend(handles = scatter.legend_elements()[0], labels=['Cancer','Sain'])
            plt.savefig(path+"/{:.0f}_UMAP_Cancer_ground_truth_{setdata}".format(num_epoch,setdata=datas)+".png")
            plt.close()
        ind=ind+1

def print_case(num):
    return{
        0: "Train",
        1: "Test",
        2: "Eval",
    }[num]  

def recup_list_donnée(extractor, discriminator, dataloader ,device=None):
    outputs, labels, discriminator_labels = [], [], []
    for batch in dataloader:
        X, Y = batch
        if device :
            X = X.to(device)
            Y = Y.to(device)
        output = extractor(X.view(-1, X.shape[1]))
        labels.extend(Y.cpu().detach().numpy().tolist())
        outputs.extend(output.cpu().detach().numpy().tolist())
    return outputs, labels

    