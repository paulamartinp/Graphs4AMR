import torch
import torch.nn.functional as F
import torch.nn as nn

class standard_gcnn_layer(nn.Module):
    def __init__(self, S, in_dim, out_dim, seed):
        super().__init__()
        torch.manual_seed(seed)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.S = S.clone()
        
        self.S += torch.eye(self.S.shape[0], device=self.S.device)
        self.d = self.S.sum(1)
        self.D_inv = torch.diag(1 / torch.sqrt(self.d))
        self.S = self.D_inv @ self.S @ self.D_inv
        self.S = nn.Parameter(self.S, requires_grad=False)
        
        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.b = nn.Parameter(torch.empty(self.out_dim))
        std = 1 / (self.in_dim * self.out_dim)
        nn.init.uniform_(self.b.data, -std, std)
        
    def forward(self, x):
        return self.S @ x @ self.W + self.b[None, :]

class standard_gcnn(nn.Module):
    def __init__(self, n_layers, dropout, hid_dim, S,
                 in_dim, out_dim, fc_layer, seed,
                 nonlin=nn.LeakyReLU):
        
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.nonlin = nonlin()
        self.n_layers = n_layers
        self.dropout = dropout

        if self.n_layers > 1:
            self.convs.append(standard_gcnn_layer(S, in_dim, hid_dim, seed))
            for _ in range(self.n_layers - 2):
                in_dim = hid_dim
                self.convs.append(standard_gcnn_layer(S, in_dim, hid_dim, seed))
            in_dim = hid_dim
            self.convs.append(standard_gcnn_layer(S, in_dim, out_dim, seed))
        else:
            self.convs.append(standard_gcnn_layer(S, in_dim, out_dim, seed))
            
        self.classifier = nn.Sequential(
            nn.Linear(fc_layer[0][0], fc_layer[0][1]),
            nn.Dropout(dropout),
        )
        self.first_layer_weights = self.classifier[0].weight
        
    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.convs[-1](x)
        importance_pre_fc = x
        embedding = x.view(x.shape[0], x.shape[1])
        
        output = self.classifier(embedding)
        pre_sigmoid = output
        output = torch.sigmoid(output)
        
        return output, importance_pre_fc, self.first_layer_weights, pre_sigmoid, self.convs[0].W, embedding



class MultiStepGNN(nn.Module):
    def __init__(self, n_layers, dropout, hid_dim, S_list,
                 in_dim, out_dim, fc_layer, seed, aggregation,
                 nonlin=nn.LeakyReLU):
        """
            n_layers (int): Number of layers in the GCN.
            dropout (float): Dropout rate.
            hid_dim (int): Hidden layer dimension.
            S_list (list): List of adjacency matrices (one per time step).
            in_dim (int): Input feature dimension.
            out_dim (int): Output dimension of each GCN.
            fc_layer (tuple): Dimensions of the final FC layer (e.g., (D, P)).
            seed (int): Random seed for reproducibility.
            nonlin (torch.nn.Module): Non-linearity function.
            aggregation (str): Method to merge embeddings ("mean", "sum", or "concat").
        """
        super().__init__()
        
        self.num_steps = len(S_list)
        self.aggregation = aggregation

        # List of GNNs for each time step
        self.gnns = nn.ModuleList([
            standard_gcnn(n_layers, dropout, hid_dim, S_list[t],
                          in_dim, out_dim, fc_layer, seed, nonlin)
            for t in range(self.num_steps)
        ])
        
        # Determine the input size for the FC layer after aggregation
        embedding_dim = out_dim  # (P, F)
        
        if self.aggregation == "concat":
            embedding_dim = self.num_steps * fc_layer[0][0]
        else:
            embedding_dim = fc_layer[0][0]
        
        # Final Fully Connected layer
        self.fc_layer = nn.Linear(embedding_dim, fc_layer[0][1])  # Use embedding_dim for input size


    def forward(self, x_list):
        """
        Forward pass of the model.

        Args:
            x_list (list): List of input features at different time steps.

        Returns:
            torch.Tensor: Final output tensor with shape (P,).
        """
        embeddings = []

        for t in range(self.num_steps):
            _, _, _, _, _, embedding = self.gnns[t](x_list[t])  # Embedding has shape (P, F)
            embeddings.append(embedding)

        # Convert list to a tensor with shape (T, P, F)
        embeddings = torch.stack(embeddings, dim=0)

        # Aggregate embeddings across T
        if self.aggregation == "mean":
            fused_embedding = torch.mean(embeddings, dim=0)  # (P, F)
        elif self.aggregation == "sum":
            fused_embedding = torch.sum(embeddings, dim=0)  # (P, F)
        elif self.aggregation == "concat":
            fused_embedding = torch.cat([emb for emb in embeddings], dim=-1)  # (P, T*F)
        else:
            raise ValueError("Invalid aggregation method. Use 'mean', 'sum', or 'concat'.")

        # Pass through the Fully Connected Layer to get final output
        output = self.fc_layer(fused_embedding)  # (P, 1)
        output = torch.sigmoid(output)
        
        return output, embeddings 
