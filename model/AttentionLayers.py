
import torch
import torch.nn as nn

from mamba_ssm import Mamba
# from model.modules.ssm import SSM

class MambaLayer(nn.Module):
    def __init__(self, embed_dim):
        super(MambaLayer, self).__init__()
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=embed_dim, # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            # headdim=4,  # Attention head dimension
            # d_model * expand / headdim = multiple of 8
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
    def forward(self, x):
        return self.mamba(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.layer = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer(x, x, x)[0]
        x = self.dropout(x)
        return x

# Define the SSM Block for the 3D model
class SSMBlock3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SSMBlock3D, self).__init__()
        # Transition matrix A, input matrix B, and observation matrix C
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # Transition matrix
        self.B = nn.Parameter(torch.randn(input_dim, hidden_dim))    # Input matrix
        self.C = nn.Parameter(torch.randn(hidden_dim, output_dim))   # Observation matrix
        
        # Non-linear activation function
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch_size, depth, height, width, input_dim)
        B, D, H, W, C = x.shape
        x = x.reshape(B * D * H * W, C)  # Flatten spatial dimensions into one sequence

        # Initialize the hidden state for each sequence
        h = torch.zeros(B * D * H * W, self.B.shape[1]).to(x.device)

        # Apply the input transformation for the entire input tensor
        h = self.activation(torch.matmul(x, self.B))  # Input transformation

        # Update the hidden state with the transition matrix A
        h = self.activation(torch.matmul(h, self.A))

        # Apply the observation matrix C to get the output
        out = torch.matmul(h, self.C)

        # Reshape output back to original spatial dimensions
        out = out.reshape(B, D, H, W, -1)
        
        return out


# Simple State Space Model (SSM) Layer
class SimpleSSMLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super(SimpleSSMLayer, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
        # basic SSM layer for now
        self.B = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
        self.C = nn.Parameter(torch.zeros(hidden_dim, embed_dim))
        nn.init.xavier_normal_(self.B)
        nn.init.xavier_normal_(self.C)
        # self.ssm = SSM(dim, dt_rank, dim_inner, d_state)
        
    def forward(self, x):
        # This is a simplified SSM; you can expand it for more complex SSM architectures
        
        h = torch.matmul(x, self.B)
        y = torch.matmul(h, self.C)
        return y
