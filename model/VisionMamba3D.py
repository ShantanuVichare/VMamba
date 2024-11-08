import torch
import torch.nn as nn

from model.AttentionLayers import MultiHeadAttention, SimpleSSMLayer, MambaLayer

# Patch Embedding for 3D input
class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbedding3D, self).__init__()
        assert len(img_size) == 3, "img_size must be a 3-tuple"
        assert len(patch_size) == 3, "patch_size must be a 3-tuple"
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # Projected patches [B, embed_dim, D', H', W']
        return x

# Custom Transformer Block
class TransformerBlockWithSSM(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, num_heads=None, mlp_ratio=4.0, dropout=0.25):
        super(TransformerBlockWithSSM, self).__init__()
        if hidden_dim is None:
            hidden_dim = int(embed_dim * mlp_ratio)
        
        # Define Layer Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        
        # Choose Attention Layer implementation
        # self.attention_layer = MultiHeadAttention(embed_dim, num_heads)
        # self.attention_layer = SimpleSSMLayer(embed_dim)
        self.attention_layer = MambaLayer(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout after GELU
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)  # Dropout after second Linear layer
        )
        
    def forward(self, x):        
        # Selected Attention layer impl + residual
        x = self.norm1(x + self.attention_layer(x))
        
        # Feedforward
        x = self.mlp(x)

        # # MLP block with residual connection
        # x = self.norm2(x + self.mlp(x))

        return x

# Vision Mamba Model
class VisionMamba3D(nn.Module):
    def __init__(self,
                 img_size, patch_size,
                 in_chans, num_classes,
                 depths=[2, 2, 6, 2],
                 dims=[96, 192, 384, 768],
                 mlp_ratio=4.0, dropout=0.5,
                 debug=False
                 ):
        super(VisionMamba3D, self).__init__()
        assert len(depths) == len(dims), "depths and dims must have the same length"
        self.num_stages = len(depths)
        self.debug = debug
        
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_chans, dims[0])
        
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
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, x):
        # x.shape = [B, in_chans, D, H, W]
        
        # Patch embedding
        x = self.patch_embed(x) # Projected patches [B, embed_dim, D', H', W']
        
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
        
        # Final bottleneck layer
        x = self.bottleneck(x) # [B, 1, D', H', W']

        # Global average pooling
        x = x.mean(dim=[2, 3, 4])  # Pool over spatial dimensions
        
        # Classification
        x = self.fc(x)

        return x

