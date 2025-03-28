import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

import modelsvf as models

###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################

def saveBestHyperparameters(best_hyperparameters, filename):
    """
    Save the best hyperparameters to a JSON file.
    
    Args:
        best_hyperparameters (dict): The dictionary containing the best hyperparameters.
        filename (str): The name of the file to save the hyperparameters.
    """
    with open(filename, 'w') as file:
        json.dump(best_hyperparameters, file, indent=4)
        
def loadBestHyperparameters(filename):
    """
    Load the best hyperparameters from a JSON file.
    
    Args:
        filename (str): The name of the file to load the hyperparameters from.
    
    Returns:
        dict: The dictionary containing the loaded hyperparameters.
    """
    with open(filename, 'r') as file:
        hyperparameters = json.load(file)
        
    return hyperparameters

#########################
# Function to load data #
#########################

def load_data(norm, device, split, numberOfTimeStep=14, time_step=0):
    """
    Load and preprocess the training, validation, and test data.
    
    Args:
        norm (str): Normalization type.
        device (torch.device): Device to load the data onto (CPU or GPU).
        split (str): Data split name (e.g., 'train', 'val', 'test').
        SG (bool): If True, applies special graph (SG) preprocessing. Default is False.
        numberOfTimeStep (int): Number of time steps to consider. Default is 14.
        time_step (int): Specific time step to extract. Default is 0 (first time step).
    
    Returns:
        tuple: Preprocessed tensors for training, validation, and test sets, along with their labels.
    """
    # Load raw data
    X_train = np.load(f"../../DATA/{split}/X_train_tensor_{norm}.npy")
    X_val = np.load(f"../../DATA/{split}/X_val_tensor_{norm}.npy")
    X_test = np.load(f"../../DATA/{split}/X_test_tensor_{norm}.npy")
    
    y_train = pd.read_csv("../../DATA/" + str(split) + "/y_train_tensor_"+norm+".csv")[['individualMRGerm_stac']].individualMRGerm_stac.values

    y_val = pd.read_csv("../../DATA/" + str(split) + "/y_val_tensor_"+norm+".csv")[['individualMRGerm_stac']].individualMRGerm_stac.values
    
    y_test = pd.read_csv("../../DATA/" + str(split) + "/y_test_tensor_"+norm+".csv")[['individualMRGerm_stac']].individualMRGerm_stac.values
    
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # print(y_train.shape)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Handle missing values and extract the specified time step
    X_train[X_train == 666] = 0
    X_val[X_val == 666] = 0
    X_test[X_test == 666] = 0

#     # Validate the time_step index
#     if time_step < 0 or time_step >= numberOfTimeStep:
#         raise ValueError(f"Invalid time_step: {time_step}. Must be between 0 and {numberOfTimeStep - 1}.")
    
    # Select the specific time step
    X_train = X_train[:, time_step, :].reshape(-1, X_train.shape[2], 1)
    X_val = X_val[:, time_step, :].reshape(-1, X_val.shape[2], 1)
    X_test = X_test[:, time_step, :].reshape(-1, X_test.shape[2], 1)
    
    # Convert to tensors
    X_train_vec = torch.tensor(X_train, dtype=torch.float32)
    X_val_vec = torch.tensor(X_val, dtype=torch.float32)
    X_test_vec = torch.tensor(X_test, dtype=torch.float32)
    
    # Move to device if specified
    if device.type == "cuda":
        return (X_train_vec.to(device), X_val_vec.to(device), X_test_vec.to(device), 
                y_train.to(device), y_val.to(device), y_test.to(device))
    else:
        return X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test


    
############################
# Function to train models #
############################

def train(model, X, y, optimizer, loss_function):
    """
    Perform a single training step for the model.
    
    Args:
        model (torch.nn.Module): The GCN model to train.
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        loss_function (torch.nn.Module): Loss function to use.
    
    Returns:
        torch.Tensor: The computed loss for the current training step.
    """
    # Mark the GCN model as being in training mode
    model.train()
    # Clear the previously calculated gradients
    optimizer.zero_grad() 
    
    # Perform forward pass
    output, embeddings, pre_sigmoid = model(X)


    output = output.squeeze()
    """
    Find the corresponding output probability and label according to the data index of the training set, 
    then calculate the loss
    """
    loss = loss_function(output, y)      
    # Perform backpropagation
    loss.backward() 
    # Update the model parameters
    optimizer.step()
    
    return loss


def train_val_phase(A, X_train, X_val, y_train, y_val, params, device):
    """
    Train the model and validate on the validation set to find the best hyperparameters.
    
    Args:
        A (torch.Tensor): Graph adjacency matrix.
        X_train (torch.Tensor): Training data.
        X_val (torch.Tensor): Validation data.
        y_train (torch.Tensor): Training labels.
        y_val (torch.Tensor): Validation labels.
        params (dict): Dictionary of training parameters and hyperparameters.
        device (torch.device): Device to run the training on (CPU or GPU).
    
    Returns:
        dict: The best hyperparameters found during training.
    """
    # Initialize
    n_epochs = params['n_epochs']
    # Early stopping configuration
    early_stopping_patience = params['early_stopping_patience']
    
    # Hyperparameters to optimize
    h_dropout = params['h_dropout']
    h_learning_rate = params['h_learning_rate']
    h_decay = params['h_decay']
    h_hid_lay = params['h_hid_lay']
    h_K = params['K']
    h_layers = params['h_layers']
    h_seed = params['seed']
    fc_layer = params['fc_layer']

    aggregation = params['aggregation']

    loss_function = torch.nn.BCELoss()

    
    bestHyperparameters = {'dropout': -1, 'decay': -1, 'lr': -1, 'hid_lay': -1, 'K': -1, 'seed': -1}
    bestMetricDev = float('inf')

    for d in range(len(h_dropout)):
        for lr in range(len(h_learning_rate)):
            for dec in range(len(h_decay)):
                for hl in range(len(h_hid_lay)):
                    for k in range(len(h_K)):
                        for s in range(len(h_seed)):
                            for l in range(len(h_layers)):

                                torch.cuda.empty_cache()

                                best_validation_loss = float('inf')
                                counter_early_stop = 0

                                n_layers = h_layers[l]
                                dropout = h_dropout[d]
                                hid_dim = h_hid_lay[hl]
                                in_dim = params['in_dim_GCN']
                                out_dim = params['out_dim_GCN']

                                
                                if params['typeGCN'] == "standard_gcnn":
                                    model = models.MultiStepGNN(n_layers, dropout, hid_dim, A,
                                                            in_dim, out_dim,
                                                            fc_layer,
                                                            h_seed[s],
                                                            aggregation,
                                                            nonlin=nn.LeakyReLU
                                                            ).to(device)

                                else:
                                    raise ValueError("ERROR! Define GCNbankFilter or GCNLPF.")

                                optimizer = torch.optim.Adam(model.parameters(), 
                                                             lr=h_learning_rate[lr], 
                                                             weight_decay=h_decay[dec])

                                for epoch in range(n_epochs):
                                    train_loss = train(model, X_train, y_train, optimizer, loss_function)

                                    """Use the validation set data to verify the epoch training results. 
                                    The verification process needs to close the train mode and open the eval model"""
                                    model.eval()
                                    # Same forward propagation
                                    with torch.no_grad():
                                        output, embeddings, pre_sigmoid = model(X_val)
                                        
                                        output = output.squeeze()
                                        """
                                        Find the corresponding output probability and label according to the data index of the validation set,
                                        then calculate the loss
                                        """
                                        loss_val = loss_function(output, y_val)

                                        # Early stopping
                                        if loss_val < best_validation_loss:
                                            best_validation_loss = loss_val
                                            counter_early_stop = 0
                                        else:
                                            counter_early_stop += 1

                                        if counter_early_stop >= early_stopping_patience:
                                            print(f'Early stopping at epoch {epoch}')
                                            break
                                            
                                if best_validation_loss < bestMetricDev:
                                    bestMetricDev = best_validation_loss
                                    bestHyperparameters['dropout'] = d
                                    bestHyperparameters['decay'] = dec
                                    bestHyperparameters['lr'] = lr
                                    bestHyperparameters['hid_lay'] = hl
                                    bestHyperparameters['K'] = k
                                    bestHyperparameters['n_lay'] = l
                                    bestHyperparameters['seed'] = s
                                    bestHyperparameters['best_loss'] = bestMetricDev.item()

    return bestHyperparameters, embeddings


def val_model(best_result_by_split, typeOfGraph, params, folders, norm, device, path_A, way_to_build_graph, T, seed):
    """
    Validate the model on the test set using the best hyperparameters.

    Args:
        best_result_by_split (dict): Best hyperparameters for each split.
        typeOfGraph (str): Type of graph to use.
        params (dict): Dictionary of training parameters and hyperparameters.
        folders (list): List of data folders.
        norm (str): Normalization type.
        device (torch.device): Device to run the validation on (CPU or GPU).
        path_A (str): Path to the adjacency matrix.
        way_to_build_graph (str): Method to build the graph.
        path
        

    Returns:
        tuple: Results, interpretability data, fully connected classifiers, and GNN models.
    """
    results = {'test_acc': [], 'roc_auc': [], 'sensitivity': [], 'specificity': []}
    # Initialize 
    n_epochs = params['n_epochs']
    # Early stopping configuration
    early_stopping_patience = params['early_stopping_patience']
    interpretability = []
    loss_function = torch.nn.BCELoss()

    fc_classifiers = {}
    gnn_models = {}
    embeddings = []

    for folder in range(len(folders)):
        torch.cuda.empty_cache()
        best_validation_loss = float('inf')
        counter_early_stop = 0

        bestHyperparameters = best_result_by_split[folders[folder]]
        
        S_list = []  # List of adjacency matrices
        X_list = []  # List of training feature matrices
        X_val_list = []  # List of validation feature matrices
        X_test_list = []
        
        for t in range(T):
            
            torch.cuda.empty_cache()
            
            # Load graph node features & labels
            X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test = load_data(
                norm, device, folders[folder], T, t
            )
    
            A = pd.read_csv(f"../dtw_matrices/{folders[folder]}/tr_AMR_{norm}_sparse.csv")
            A = torch.tensor(A.values, dtype=torch.float32)
    
            # Store adjacency matrix and node features for MultiStepGNN
            S_list.append(A)  
            X_list.append(X_train_vec)  
            X_val_list.append(X_val_vec)  
            X_test_list.append(X_test_vec)

        # X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test = load_data(norm, device, folders[folder], t)
        # A = pd.read_csv(path_A)
        
        # A = torch.tensor(np.array(A), dtype=torch.float32)
 
        dropout = params['h_dropout'][bestHyperparameters['dropout']]
        hid_dim = params['h_hid_lay'][bestHyperparameters['hid_lay']]
        n_layers = params['h_layers'][bestHyperparameters['n_lay']]
        # seed = params['seed'][bestHyperparameters['seed']]
        seed = seed
        bestK = params['K'][bestHyperparameters['K']]
        fc_layer = params['fc_layer']

        aggregation = params['aggregation']
        
        in_dim = params['in_dim_GCN']
        out_dim = params['out_dim_GCN']
        

        if params['typeGCN'] == "higher_order_polynomial_gcnn": 
            model = models.higher_order_polynomial_gcnn(n_layers, dropout, hid_dim, S_list,
                                            in_dim, out_dim, bestK, 
                                            fc_layer,
                                            seed,
                                            ).to(device)
            
        elif params['typeGCN'] == "standard_gcnn":
            model = models.MultiStepGNN(n_layers, dropout, hid_dim, S_list,
                                    in_dim, out_dim,
                                    fc_layer,
                                    seed,
                                    aggregation, nonlin=nn.LeakyReLU
                                    ).to(device)
        else:
            print("ERROR!")
                                        

        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=params['h_learning_rate'][bestHyperparameters['lr']], 
                                     weight_decay=params['h_decay'][bestHyperparameters['decay']])
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            train_loss = train(model, X_list, y_train, optimizer, loss_function)
            # Save loss by epoch
            train_losses.append(train_loss)

            """Use the validation set data to verify the epoch training results. 
            The verification process needs to close the train mode and open the eval model"""
            model.eval()
            # Same forward propagation
            with torch.no_grad():
                output, embeddings, pre_sigmoid = model(X_val_list)

                output = output.squeeze()
                """
                Find the corresponding output probability and label according to the data index of the validation set,
                then calculate the loss
                """
                loss_val = loss_function(output, y_val)
                val_losses.append(loss_val)

                # Early stopping
                if loss_val < best_validation_loss:
                    best_validation_loss = loss_val
                    counter_early_stop = 0
                else:
                    counter_early_stop += 1

                if counter_early_stop >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # fc_classifiers[folders[folder]] = model.get_classifier()
        gnn_models[folders[folder]] = model
        
        model.eval()
        with torch.no_grad():  # Deactivate gradient computation
            output_final, embeddings_final, pre_sigmoid = model(X_test_list)
            
            output_final = output_final.squeeze()
            
            pred = torch.round(output_final).view(-1)


            # Check against ground-truth labels.
            test_correct = pred == y_test  
            # Derive ratio of correct predictions.
            test_acc = int(test_correct.sum()) / int(test_correct.shape[0])  
            # Calculate ROC-AUC
            roc_auc = roc_auc_score(y_test.cpu().numpy(), output_final.cpu().numpy())
            # Calculate confusion matrix for sensitivity and specificity
            tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), pred.cpu().numpy()).ravel()

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            results['test_acc'].append(test_acc)
            results['roc_auc'].append(roc_auc)
            results['sensitivity'].append(sensitivity)
            results['specificity'].append(specificity)
            
    return results, output_final, gnn_models, embeddings_final


def typeOfGNN(folders, K, best_result_by_split, importance_nodes):
    """
    Identify the type of GNN (Low Pass Filter or High Pass Filter) for each folder.
    
    Args:
        folders (list): List of data folders.
        K (list): List of filter orders.
        best_result_by_split (dict): Best hyperparameters for each split.
        importance_nodes (list): List of importance data from the GNN models.
    """
    for i in range(len(folders)):
        W = importance_nodes[i][3][0]
        S = importance_nodes[i][3][1]
        if torch.any(S > 0):
            print("Low Pass Filter.", "# of filters:", K[best_result_by_split[folders[i]]['K']])
        else:
            print("High Pass Filter.")