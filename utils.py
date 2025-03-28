import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import precision_recall_curve, auc
from matplotlib.pyplot import figure, text
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

def reshape_patients_by_features(data, keys, numberOfTimeStep):
    """
    Reshapes the data based on the number of features and time steps.
    
    Parameters:
    data (ndarray): The original data array.
    keys (list): List of feature names.
    numberOfTimeStep (int): The number of time steps.
    
    Returns:
    ndarray: The reshaped data.
    """
    df = pd.DataFrame(data.reshape(int(data.shape[0]*numberOfTimeStep), data.shape[2]))
    df.replace(666, -1, inplace=True)
    df.columns = keys

    for i in range(len(keys)):
        df_trial = df[keys[i]]
        if i == 0:
            X = np.array(df_trial)
            X = X.reshape((int(df_trial.shape[0]/numberOfTimeStep), numberOfTimeStep)).T
            X = X.reshape(1, numberOfTimeStep, int(df_trial.shape[0]/numberOfTimeStep))

        else:
            X_2 = np.array(df_trial)
            X_2 = X_2.reshape((int(df_trial.shape[0]/numberOfTimeStep), numberOfTimeStep)).T
            X_2 = X_2.reshape(1, numberOfTimeStep, int(df_trial.shape[0]/numberOfTimeStep))

            X = np.append(X, X_2, axis=0)
        
    return X


def exp_kernel(train, sigma):
    """
    Exponential kernel function that computes a kernel matrix based on a dataset and a sigma parameter.
    
    Args:
        train (pd.DataFrame): A dataset where rows represent data points.
        sigma (float): A scaling parameter for the kernel.
    
    Returns:
        pd.DataFrame: The kernel matrix with values computed based on the exponential distance between data points.
    """
    #     matrix_train = np.zeros((train.shape[0],train.shape[0]))
    #     for i in range(train.shape[0]):
    #         for j in range(train.shape[0]):
    #             matrix_train[i,j] = np.exp(-((np.linalg.norm(train.iloc[i,:].values - train.iloc[j,:].values, ord=2, axis=0))/(2*(sigma**2))))


    matrix_train = np.zeros((train.shape[0], train.shape[0]))
    
    for i in range(train.shape[0]):
        for j in range(train.shape[0]):
            matrix_train[i, j] = np.exp(-((np.linalg.norm(train.iloc[i, :].values - train.iloc[j, :].values, ord=2, axis=0)) / (2 * (sigma ** 2))))

    matrix_train = np.exp(-(train ** 2) / (2 * (sigma ** 2)))

    x = pd.DataFrame(matrix_train)
    x = np.round(x, 6)

    print(x.loc[0])

    return x


def edge_density_entropy(data):
    """
    Computes edge density and edge entropy for a given dataset.
    
    Args:
        data (pd.DataFrame): A dataset with numeric values.
    
    Returns:
        tuple: A tuple containing edge density and edge entropy values.
    """
    num_f = data.shape[0]
    df = (1 / num_f) * np.sum(np.abs(data)).values

    edge_density = (1 / (num_f - 1)) * np.sum(df)
    edge_entropy = 0
    for i in range(num_f):
        edge_entropy += df[i] * np.log(df[i])
    edge_entropy = edge_entropy * -1

    return edge_density, edge_entropy


def label_rows(list_of_lists):
    """
    Labels each row of a list of lists based on the position of the minimum value.
    
    Args:
        list_of_lists (list of lists): A list where each element is a list of numeric values.
    
    Returns:
        list: A list of labels, where 1 indicates the first element is the minimum, and 0 otherwise.
    """
    labels = []  # This list will store the labels for each row
    
    for row in list_of_lists:
        if len(row) == 0:
            labels.append("No elements in the row")  # Handle empty rows
        else:
            min_value = min(row)
            if row[0] == min_value:
                labels.append(1)
            else:
                labels.append(0)
    
    return labels


def calculate_auprc(tp, fp, tn, fn):
    """
    Calculates the Area Under the Precision-Recall Curve (AUPRC) using true positives, false positives, true negatives, and false negatives.
    
    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        tn (int): True Negatives.
        fn (int): False Negatives.
    
    Returns:
        float: The AUPRC value.
    """
    y_true = [1] * (tp + fn) + [0] * (fp + tn)
    y_scores = [1] * tp + [0] * fp + [0] * tn + [1] * fn

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    auprc = auc(recall, precision)

    return auprc


def calculate_f1_score(tp, fp, fn):
    """
    Calculates the F1 score, which is the harmonic mean of precision and recall.
    
    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.
    
    Returns:
        float: The F1 score.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def get_metrics(y, result_by_pat, metric):
    """
    Computes various performance metrics (ROC-AUC, Sensitivity, Specificity, AUPRC, F1 Score) based on the given true labels and predictions.
    
    Args:
        y (pd.DataFrame): A DataFrame containing true labels for comparison.
        result_by_pat (list of lists): A list of lists containing the prediction results.
        metric (str): The metric to calculate ("ROC-AUC", "Sensitivity", "Specificity", "AUPRC", "f1score").
    
    Returns:
        tuple: The computed metric value and the updated DataFrame with predictions.
    """
    labels = label_rows(result_by_pat)
    y['y_pred'] = labels

    TP = y[(y.individualMRGerm_stac == 1.0) & (y.y_pred == 1.0)].shape[0]
    FP = y[(y.individualMRGerm_stac == 1.0) & (y.y_pred == 0.0)].shape[0]
    FN = y[(y.individualMRGerm_stac == 0.0) & (y.y_pred == 1.0)].shape[0]
    TN = y[(y.individualMRGerm_stac == 0.0) & (y.y_pred == 0.0)].shape[0]

    if metric == "ROC-AUC":
        try:
            TPR = TP / (TP + FN)
        except ZeroDivisionError:
            TPR = 0
        try:
            FPR = FP / (FP + TN)
        except ZeroDivisionError:
            FPR = 0
        val_metric = np.round(100 * (1 + TPR - FPR) / 2, 4)
    elif metric == "Sensitivity":
        try:
            val_metric = TP / (TP + FN)
        except ZeroDivisionError:   
            val_metric = 0
    elif metric == "Specificity":
        try:
            val_metric = TN / (TN + FP)
        except ZeroDivisionError:
            val_metric = 0
    elif metric == "AUPRC":
        val_metric = calculate_auprc(TP, FP, TN, FN)
    elif metric == "f1score":
        val_metric = calculate_f1_score(TP, FP, FN)
        
    return val_metric, y


def build_and_export_graph(shift_operator, params):
    """
    Builds a graph from a shift operator matrix and the parameters. The edges of the graph are weighted by the shift operator values.
    
    Args:
        shift_operator (np.ndarray or pd.DataFrame): A matrix representing relationships between nodes.
        params (list): A list of parameter names corresponding to the nodes of the graph.
    
    Returns:
        networkx.Graph: A graph with edges weighted by the shift operator values.
    """
    aux = pd.DataFrame(data=shift_operator, columns=params)
    aux.index = params

    G = nx.Graph()
    
    for i in range(aux.shape[0]):
        for j in range(aux.shape[0]):
            G.add_edges_from([(params[i], params[j])], weight=aux.iloc[i, j])
            
    return G



##################################################
##### Node classification specific functions #####
##################################################
def graph_function(df, threshold_val):
    """
    Applies a threshold to a covariance matrix and constructs a graph from the resulting matrix.

    Args:
        df (pd.DataFrame): The covariance matrix.
        threshold_val (float): Threshold value to set small values to zero.
    
    Returns:
        tuple: 
            - GDead (networkx.Graph): The graph built from the thresholded covariance matrix.
            - s (pd.DataFrame): The modified covariance matrix after applying the threshold.
    """
    print("Covariance matrix size:", df.shape)
    print("Number of non-zero values before applying the threshold:", np.count_nonzero(df))

    s = df.copy()
    s[np.abs(s) < threshold_val] = 0  # Set values below the threshold to zero
    print("Number of non-zero values after applying the threshold:", np.count_nonzero(s))

    keys = df.keys()
    
    # Build the graph using the thresholded matrix
    GDead = build_and_export_graph(np.abs(s), keys)
    
    return GDead, s


def get_process_data(X, y):
    """
    Processes raw data into PyTorch tensor format.
    
    Args:
        X (np.ndarray): Raw data in a 3D array (patients x time series x features).
        y (pd.DataFrame): A DataFrame containing target labels.
    
    Returns:
        tuple: 
            - feats (torch.Tensor): A tensor of features for each patient.
            - labels (torch.Tensor): A tensor of labels corresponding to each patient.
    """
    feats = []
    labels = []

    for p in range(X.shape[0]):
        values_by_pat = []
        
        for f in range(X.shape[2]):
            mask = (X[p, :, f] != 666)  # Ignore missing values (encoded as 666)
            values = X[p, :, f][mask]  # Extract valid values
            values_by_pat.append(np.mean(values))  # Calculate the mean of valid values

        feats.append(torch.tensor(values_by_pat, dtype=torch.float32))
        labels.append(y['individualMRGerm_stac'].loc[p])  # Get corresponding label for patient

    # Stack the features into a single tensor
    feats = torch.stack(feats, dim=0) 
    labels = torch.tensor(labels)

    return feats, labels
