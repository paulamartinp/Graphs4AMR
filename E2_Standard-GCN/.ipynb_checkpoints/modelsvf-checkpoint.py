import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple

class StandardGCNNLayer(nn.Module):
    """
    Standard Graph Convolutional Neural Network (GCN) layer.

    This layer applies graph convolution using a normalized adjacency matrix.

    Parameters
    ----------
    S : torch.Tensor
        Normalized adjacency matrix of shape (N, N).
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    seed : int
        Random seed for weight initialization.
    """

    def __init__(self, S: torch.Tensor, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.S = S.clone()
        
        # Add self-loops
        self.S += torch.eye(self.S.shape[0], device=self.S.device)
        
        # Degree normalization
        self.d = self.S.sum(1)
        eps = 1e-6  # Small constant to avoid division by zero
        self.D_inv = torch.diag(1 / torch.sqrt(self.d + eps))
        self.S = self.D_inv @ self.S @ self.D_inv
        self.S = nn.Parameter(self.S, requires_grad=False)
        
        # Initialize weights and bias
        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.b = nn.Parameter(torch.empty(self.out_dim))
        std = 1 / (self.in_dim * self.out_dim)
        nn.init.uniform_(self.b.data, -std, std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (N, in_dim).

        Returns
        -------
        torch.Tensor
            Output features of shape (N, out_dim).
        """
        return self.S @ x @ self.W + self.b[None, :]


class StandardGCNN(nn.Module):
    """
    Multi-layer Graph Convolutional Neural Network (GCN).

    Parameters
    ----------
    n_layers : int
        Number of GCN layers.
    dropout : float
        Dropout probability.
    hid_dim : int
        Hidden layer dimension.
    S : torch.Tensor
        Normalized adjacency matrix.
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    seed : int
        Random seed for reproducibility.
    nonlin : nn.Module, optional
        Non-linearity function (default: LeakyReLU).
    """

    def __init__(self, n_layers: int, dropout: float, hid_dim: int, S: torch.Tensor,
                 in_dim: int, out_dim: int, seed: int, nonlin: nn.Module = nn.LeakyReLU):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.nonlin = nonlin()
        self.n_layers = n_layers
        self.dropout = dropout

        if self.n_layers > 1:
            self.convs.append(StandardGCNNLayer(S, in_dim, hid_dim, seed))
            for _ in range(self.n_layers - 2):
                in_dim = hid_dim
                self.convs.append(StandardGCNNLayer(S, in_dim, hid_dim, seed))
            in_dim = hid_dim
            self.convs.append(StandardGCNNLayer(S, in_dim, out_dim, seed))
        else:
            self.convs.append(StandardGCNNLayer(S, in_dim, out_dim, seed))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer GCN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (N, in_dim).

        Returns
        -------
        torch.Tensor
            Output embeddings of shape (N, out_dim).
        """
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.convs[-1](x)
        embedding = x.view(x.shape[0], x.shape[1])  # Flatten embeddings if necessary
        
        return embedding


class MultiStepGNN(nn.Module):
    """
    Multi-step GNN that processes multiple temporal snapshots and combines embeddings.

    Parameters
    ----------
    n_layers : int
        Number of GCN layers.
    dropout : float
        Dropout probability.
    hid_dim : int
        Hidden layer dimension.
    S_list : list of torch.Tensor
        List of adjacency matrices for each time step.
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    fc_layer : list of tuple (int, int)
        Fully connected layer dimensions.
    seed : int
        Random seed for reproducibility.
    nonlin : nn.Module, optional
        Non-linearity function (default: LeakyReLU).
    combine_method : str, optional
        Method to combine embeddings across time steps. Options: "concat", "sum", "mean".
        Default is "concat".
    """

    def __init__(self, n_layers: int, dropout: float, hid_dim: int, S_list: List[torch.Tensor],
                 in_dim: int, out_dim: int, fc_layer: List[Tuple[int, int]], seed: int, 
                combine_method: str = "concat", nonlin: nn.Module = nn.LeakyReLU):
        super().__init__()

        self.num_steps = len(S_list)
        self.combine_method = combine_method

        self.gnns = nn.ModuleList([
            StandardGCNN(n_layers, dropout, hid_dim, S_list[t],
                         in_dim, out_dim, seed, nonlin)
            for t in range(self.num_steps)
        ])
        
        # Adjust input dimension of the fully connected layer based on the combination method
        fc_input_dim = fc_layer[0][0] * self.num_steps if self.combine_method == "concat" else fc_layer[0][0]

        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, fc_layer[0][1]),
            nn.Dropout(dropout)
        )

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MultiStepGNN.

        Parameters
        ----------
        x_list : list of torch.Tensor
            List of node feature matrices, one per time step.

        Returns
        -------
        tuple
            output : torch.Tensor
                Final classification output of shape (batch_size, 1).
            combined_embedding : torch.Tensor
                Combined node embeddings.
            pre_sigmoid : torch.Tensor
                Raw logits before sigmoid activation.
        """
        embeddings = []

        for t in range(self.num_steps):
            embedding = self.gnns[t](x_list[t])  # Obtain embedding from each GNN
            embeddings.append(embedding)

        # Combine embeddings based on the chosen method
        if self.combine_method == "concat":
            combined_embedding = torch.cat(embeddings, dim=1) if self.num_steps > 1 else embeddings[0]
        elif self.combine_method == "sum":
            combined_embedding = torch.stack(embeddings, dim=0).sum(dim=0)
        elif self.combine_method == "mean":
            combined_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        else:
            raise ValueError("combine_method must be 'concat', 'sum', or 'mean'.")

        # Pass the combined embedding through the classifier
        output = self.classifier(combined_embedding)
        pre_sigmoid = output
        output = torch.sigmoid(output)

        return output, combined_embedding, pre_sigmoid
