import torch
import torch.nn as nn
from mamba_ssm import Mamba
# from model.modules.ssm import SSM

# Patch Embedding for 3D input
# class PatchEmbedding3D(nn.Module):
#     def __init__(self, img_size, patch_size, in_chans, embed_dim):
#         super(PatchEmbedding3D, self).__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
#         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#     def forward(self, x):
#         x = self.proj(x)  # Projected patches [B, embed_dim, D', H', W']
#         x = x.flatten(2)  # Flattened patches [B, embed_dim, num_patches]
#         x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]
#         return x

# Custom Transformer Block with State Space Model (SSM) layer
class TransformerBlockWithSSM(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, num_heads=None, mlp_ratio=4.0, dropout=0.5):
        super(TransformerBlockWithSSM, self).__init__()
        if hidden_dim is None:
            hidden_dim = int(embed_dim * mlp_ratio)
        # self.norm = nn.LayerNorm(embed_dim)
        
        # Define 6 Layer Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.norm5 = nn.LayerNorm(embed_dim)
        self.norm6 = nn.LayerNorm(embed_dim)
        
        # self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.ssm_layer = StateSpaceModelLayer(embed_dim)
        self.mamba_layer = MambaLayer(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout after GELU
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)  # Dropout after second Linear layer
        )
        
    def forward(self, x):        
        # Self-attention
        # attn_output, _ = self.attn(x, x, x)
        # x = self.norm1(x + attn_output)
        
        # State Space Model Layer
        # ssm_output = self.ssm_layer(x)
        # x = self.norm2(x + ssm_output)
        
        # # Mamba Layer
        # mamba_output = self.mamba_layer(x)
        # x = self.norm(x + mamba_output)
        
        # # Feedforward
        # x = self.mlp(x)
        
        
        # Layer Normalization and Self-Attention with Residual Connection
        x = x + self.mamba_layer(self.norm1(x))

        # Further normalization before the MLP block
        x = self.norm4(x)

        # MLP block with residual connection
        x = x + self.mlp(self.norm5(x))

        # Final normalization before output
        x = self.norm6(x)
        return x

# Custom State Space Model (SSM) Layer
# class StateSpaceModelLayer(nn.Module):
#     def __init__(self, embed_dim, hidden_dim=None):
#         super(StateSpaceModelLayer, self).__init__()
#         if hidden_dim is None:
#             hidden_dim = embed_dim * 2
#         # basic SSM layer for now
#         self.B = nn.Parameter(torch.zeros(embed_dim, hidden_dim))
#         self.C = nn.Parameter(torch.zeros(hidden_dim, embed_dim))
#         nn.init.xavier_normal_(self.B)
#         nn.init.xavier_normal_(self.C)
#         # self.ssm = SSM(dim, dt_rank, dim_inner, d_state)
        
#     def forward(self, x):
#         # This is a simplified SSM; you can expand it for more complex SSM architectures
        
#         h = torch.matmul(x, self.B)
#         y = torch.matmul(h, self.C)
#         return y
    
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

# Vision Mamba Model
class VisionMamba3D(nn.Module):
    def __init__(self,
                 img_size=(155, 240, 240), patch_size=(5, 8, 8),
                 in_chans=1, num_classes=2,
                 depths=[2, 2, 6, 2],
                 dims=[96, 192, 384, 768],
                 mlp_ratio=4.0, dropout=0.5,
                 debug=False
                 ):
        super(VisionMamba3D, self).__init__()
        assert len(depths) == len(dims), "depths and dims must have the same length"
        self.num_stages = len(depths)
        self.debug = debug
        
        # self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_chans, dims[0])
        self.patch_embed = nn.Conv3d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size)
        
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(*[
                TransformerBlockWithSSM(
                    embed_dim=dims[i_layer],
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(depths[i_layer]) ])
            for i_layer in range(self.num_stages)
        ])
        self.downsample_layers = nn.ModuleList([
            nn.Conv3d(dims[i_layer], dims[i_layer+1], kernel_size=(2, 2, 2), stride=(2, 2, 2))
            for i_layer in range(self.num_stages-1)
        ])
        
        # Final bottleneck layer
        self.bottleneck = nn.Conv3d(dims[-1], dims[-1], kernel_size=1)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        # x.shape = [B, in_chans, D, H, W]
        
        # Patch embedding
        x = self.patch_embed(x) # Projected patches [B, embed_dim, D', H', W']
        if self.debug: print('NaN check - post patch embed', torch.isnan(x).any())
        
        # Alternating Transformer and Downsample blocks
        for i_layer in range(self.num_stages):
            x = x.permute(0, 2, 3, 4, 1) # Permute to (B, D', H', W', embed_dim)
            x_shape = x.shape # Store shape for reshaping later (B, D', H', W', embed_dim)
            x = x.reshape(x_shape[0], -1, x_shape[-1]) # Reshape to (B, num_patches, embed_dim)
            x = self.transformer_layers[i_layer](x)
            x = x.reshape(x_shape) # Reshape back to (B, D', H', W', embed_dim)
            x = x.permute(0, 4, 1, 2, 3) # Permute back to (B, embed_dim, D', H', W')
            if i_layer < self.num_stages - 1:
                x = self.downsample_layers[i_layer](x)
        
        if self.debug: print('NaN check - post transformer layers', torch.isnan(x).any())
        
        # Final bottleneck layer
        x = self.bottleneck(x) # [B, 1, D', H', W']

        # Global average pooling
        x = x.mean(dim=[2, 3, 4])  # Pool over spatial dimensions
        
        if self.debug: print('NaN check - post bottleneck+average', torch.isnan(x).any())
        
        # Classification
        x = self.fc(x)
        
        if self.debug: print('NaN check - post final fc', torch.isnan(x).any())
        return x

# # Initialize model
# model = VisionMamba3D(
#     img_size=(240, 240, 155), 
#     patch_size=(4, 4, 3), 
#     in_chans=4, 
#     num_classes=2, 
#     embed_dim=96, 
#     depths=[4, 4, 4, 4], 
#     hidden_dim=128
# )

# # Sample input (batch of 2 3D images, with 4 channels)
# x = torch.randn(2, 4, 240, 240, 155)
# output = model(x)
# print(output.shape)  # Expected output shape: [2, 2]
