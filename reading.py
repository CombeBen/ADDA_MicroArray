import torch as torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import pickle
import os

#Les méthodes ci-dessous a pour objectif de simplifier le chargement des données dans les notebooks pour éviter de devoir aller lire dans le fichier csv des données car cette opération est très chronophage.
#Il existe donc 3 variantes de cette méthode : 1 pour toutes les données, une pour seulement les données patients, et une pour seulement les données CellLine.

def domain_selection(val):
    '''
    Return an object containing a numpy_array of selected data.
    Val must be int inside [1,3]
    1 : all_data
    2 : cell_data
    3 : patient_data
    
    Method will inspect if the pickle file is already created :
        if yes :
            use it to quickly load the data
        else :
            open the csv file, read it and create the pickle file for further use

    '''
    if val==1:
        print("Selecting All domain source")
        if(os.path.exists('serialized_object/all_data_MA.npy')):
            all_data = read_all_numpy()
        else:
            all_data = write_all_numpy()
        return all_data
    elif val ==2:
        print("Selection Cell domain source")
        if(os.path.exists('serialized_object/cellline_data_MA.npy')):
            cell_data = read_cell_numpy()
        else:
            cell_data = write_cell_numpy()
        print(len(cell_data))
        return cell_data
    elif val==3:
        print("Selection Patient domain source")
        if(os.path.exists('serialized_object/patient_data_MA.npy')):
            patient_data = read_patient_numpy()
        else:
            patient_data = write_patient_numpy()
        print(len(patient_data))
        return patient_data

        
def write_all_numpy():
    '''
    return an numpy_array containing all the data
    
    similar to read_all_numpy()
    but reads the csv file and then creates a pickle file
    for further use.
    '''
    print("Creating all dataset first_read_data()")
    all_data = first_read_data()
    print("Generating Pickle of all dataset for further use and optimising time")
    filename = "serialized_object/all_data_MA"
    np.save(filename,all_data)
    print("All done\nName of the variable containing all data : all_data")
    return all_data
    
def read_all_numpy():
    '''
    return an numpy_array containing all the data
    
    similar to write_all_numpy()
    but quickly load the pickle file.
    '''
    cell_data = np.load("serialized_object/all_data_MA.npy",allow_pickle=True)
    print("All done\nName of the variable containing all data : all_data")
    return all_data

        
       
        
def write_cell_numpy():
    '''
    return an numpy_array containing the cell data
    
    similar to read_all_numpy()
    but reads the csv file and then creates a pickle file
    for further use.
    '''
    print("Creating Cell Line dataset first_read_data()")
    cell_data = first_read_data(selected_type='cell line')
    print("Generating Pickle of Cell Line dataset for further use and optimising time")
    filename = "serialized_object/cellline_data_MA"
    np.save(filename,cell_data)
    print("All done\nName of the variable containing all cell data : cell_data")
    return cell_data
    
def read_cell_numpy():
    '''
    return an numpy_array containing the cell data
    
    similar to write_all_numpy()
    but quickly load the pickle file.
    '''
    cell_data = np.load("serialized_object/cellline_data_MA.npy",allow_pickle=True)
    print("All done\nName of the variable containing all cell data : cell_data")
    return cell_data
    
    
#Cette méthode est utilisée pour ecrire les données
def write_patient_numpy():
    '''
    return an numpy_array containing the patient
    
    similar to read_all_numpy()
    but reads the csv file and then creates a pickle file
    for further use.
    '''
    print("Creating Patient dataset first_read_data()")
    
    #On utilise la méthode pour créer l'objet à sérialisé pour gagner du temps lors de la prochaine utilisation
    patient_data = first_read_data(selected_type='patient')
    
    print("Generating Pickle of Cell Line dataset for further use and optimising time")
    
    filename = "serialized_object/patient_data_MA"
    np.save(filename,patient_data)
    
    print("All done\nName of the variable containing all the Patient data : patient_data")
    return patient_data

def read_patient_numpy():
    '''
    return an numpy_array containing the patient data
    
    similar to write_all_numpy()
    but quickly load the pickle file.
    '''
    patient_data = np.load("serialized_object/patient_data_MA.npy",allow_pickle=True)
    print("All done\nName of the variable containing all the data : Patient_data")
    return patient_data
        
        
#La méthode ci-dessous est appelé la première fois que le notebook souhaite lire les données.
#Cette opération étant coûteuse temporellement, on tente de la restreindre au minimum d'utilisation possible.
#De fait, elle n'est utilisée qu'une seule et unique fois et par la suite, on tentera de faire charger les objets sérialisé qui découlent de son utilisation.
def first_read_data(datapath="/home/commun/MicroArray/E-MTAB-3732/", domain_attribute=None, part=None, selected_type=None):
    #Lecture des classes
    label_df = pd.read_csv(os.path.join(datapath,"class.csv"))
    #Lecture des données
    data_df = pd.read_csv(os.path.join(datapath,"E-MTAB-3732.data2.csv"))
    
    #Ajustement des index, on drop la colonne 0 car c'est redondant avec les id d'un dataframe
    data_df = data_df.drop(columns=[data_df.columns[0]])
    #Les colonnes et lignes sont inversés donc, on prend sa transposée
    data_np = data_df.to_numpy(dtype=np.float32).T
    
    #on récupère les labels stockés dans le fichier class
    label_np = label_df["Cancer"].to_numpy()
    type_np = label_df["cell"].to_numpy()
    part_np =label_df['Characteristics.organism.part.'].to_numpy()
    
    #on organise un LabelEncoder pour transformer les labels "normal" et "cancer" en des valeurs numériques
    #on fit et transform pour appliquer le label encoder aux données. Puis on les transforment en numpy array d'un certain type : ici le type float32
    le = preprocessing.LabelEncoder()
    le.classes_=["normal","cancer"]
    label_np = le.fit_transform(label_np)
    label_np = label_np.astype(np.float32)
    
    if domain_attribute:
        le = preprocessing.LabelEncoder()
        le.classes_=['cell line','patient']
        type_np = le.fit_transform(type_np)
        type_np = type_np.astype(np.float32)
    if part:
        lp = preprocessing.LabelEncoder()
        lp.classes_= label_df['Characteristics.organism.part.'].unique()
        
    #la méthode ci-dessous permet d'isoler les différents types de donnée cell ou patient que l'on veut récupérer
    if selected_type:
        selected = (label_df["cell"] == selected_type).to_numpy()
        label_np = label_np[selected]
        data_np = data_np[selected]
    
    #On scale les données au moyen d'un StandardScaler
    scaler = preprocessing.StandardScaler()
    data_np = scaler.fit_transform(data_np)
    
    if domain_attribute:
        return np.concatenate((np.expand_dims(type_np,axis=0),np.expand_dims(label_np, axis=0),data_np.T)).T
    else : 
        return np.concatenate((np.expand_dims(label_np, axis=0),data_np.T)).T


#Création d'un custom dataset.
class DataSetMicroArray(Dataset):
    def __init__(self, inputs, labels, device=None):
        self.inputs=torch.from_numpy(inputs).to(dtype=torch.float32)
        self.labels=torch.from_numpy(labels).to(dtype=torch.float32)
        if device:
            self.inputs = self.inputs.to(device)
            self.labels = self.labels.to(device)
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index):
        label = self.labels[index]
        valeur = self.inputs[index]
        return valeur, label
    


#Méthode permettant la génération des indices utilisé pour lié les labels et les données.
#La découpe des données se fait avec le ratio suivant :
# train 70% | Test 15% | Eval 15%
def generation_indexes(data, rs=0):
    indexes = list(range(len(data)))
    train_idx, test_idx = train_test_split(indexes, test_size=0.15,train_size=None,random_state=rs)
    train_idx, val_idx = train_test_split(train_idx, test_size = 0.15 / (1- 0.15), train_size=None,random_state=rs)
    
    return(train_idx, val_idx, test_idx)

#Méthode permettant la récupération des dataloaders en fonction d'un certain batch_size
def get_dalaloaders( dataset, idx, bs=None, verbose= False):
    if verbose:
        print(f"{len(dataset)} elements in the dataset")
        
    train_idx, val_idx, test_idx = idx
    
    if bs is None:
        bs = [max(1,len(train_idx)),max(1,len(val_idx)),max(1,len(test_idx))]
    
    train_sample = SubsetRandomSampler(train_idx)
    val_sample = SubsetRandomSampler(val_idx)
    test_sample = SubsetRandomSampler(test_idx)
    
    Dload = DataLoader
    
    trainset = Dload(dataset, batch_size=bs[0],sampler = train_sample)
    valset = Dload(dataset, batch_size=bs[1],sampler=val_sample)
    testset = Dload(dataset, batch_size=bs[2],sampler=test_sample)
    
    if verbose:
        print(f"{len(train_idx)} elements in trainset")
        print(f"{len(val_idx)} elements in valset")
        print(f"{len(test_idx)} elements in testset")
    
    return (trainset, valset, testset)

#Méthode prenant des data en entrées et retourne les dataloaders de Train, Test et Eval correspondant.
#Et ce, avec une taille de batch size variable.
def arrange_data_into_dataloaders(data, base_bs=256, margin=0.5, device=None):
    dataset = DataSetMicroArray(data[:,1:],data[:,0],device)
    idx = generation_indexes(dataset,1)
    bs = base_bs
    dataloaders = get_dalaloaders(dataset,idx,[bs,bs,bs])
    input_dim = dataloaders[0].dataset.inputs.shape[1]
    return dataloaders, input_dim, dataset

def use_dataset(dataset, base_bs=256, margin = 0.5):
    idx = generation_indexes(dataset,1)
    bs = base_bs
    dataloaders = get_dalaloaders(dataset,idx,[bs,bs,bs],True)
    input_dim = dataloaders[0].dataset.inputs.shape[1]
    return dataloaders, input_dim
