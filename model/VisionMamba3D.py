
import torch
import torch.nn as nn



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

# 3D Vision Mamba Model with SSM blocks and downsampling
class VisionMamba3D(nn.Module):
    def __init__(self, img_size=(155, 240, 240), patch_size=(4, 4, 4), in_chans=1, num_classes=2, embed_dim=96, depths=[4, 4, 4, 4], hidden_dim=128):
        super(VisionMamba3D, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Embedding layer (linear projection of patches)
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # SSM layers with downsampling stages
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()  # Define separate downsample layers

        for i_layer in range(self.num_layers):
            # Add SSM blocks for this stage, pass the correct input_dim (embed_dim)
            stage = nn.Sequential(
                *[SSMBlock3D(input_dim=self.embed_dim, hidden_dim=hidden_dim, output_dim=self.embed_dim) for _ in range(depths[i_layer])]
            )
            self.layers.append(stage)

            # Add downsampling layers after each stage except the last
            if i_layer < self.num_layers - 1:
                downsample = nn.Conv3d(self.embed_dim, self.embed_dim * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                self.downsamples.append(downsample)
                self.embed_dim *= 2  # Update embedding dimension for the next layer

        # Final bottleneck layer
        self.bottleneck = nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=1)

        # Final MLP classifier based on bottleneck features
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Output size should be (B, embed_dim, H/patch, W/patch, D/patch)
        print('NaN check - post patch embed', torch.isnan(x).any())

        # Reshape to (B, D, H, W, C) format for SSM layers
        B, C, H, W, D = x.shape
        print(x.shape)
        x = x.permute(0, 4, 2, 3, 1)  # Permute to (B, D, H, W, C)

        # SSM blocks with downsampling
        for i_layer, layer in enumerate(self.layers):
            x = layer(x) # Apply SSM blocks
            print(f'NaN check - post SSM {i_layer+1}', torch.isnan(x).any())
            print(f'Max check - post SSM {i_layer+1}', torch.max(x))
            if i_layer < self.num_layers - 1:
                x = x.permute(0, 4, 1, 2, 3)  # Permute back to (B, C, D, H, W) for downsampling
                x = self.downsamples[i_layer](x)  # Apply downsampling
                x = x.permute(0, 2, 3, 4, 1)  # Permute back to (B, D, H, W, C)
                print(f'NaN check - post downsample {i_layer+1}', torch.isnan(x).any())
                print(f'Max check - post downsample {i_layer+1}', torch.max(x))

        # Bottleneck feature extraction
        x = x.permute(0, 4, 1, 2, 3)  # Permute to (B, C, D, H, W)
        x = self.bottleneck(x)
        print('NaN check - post bottleneck', torch.isnan(x).any())
        print('Max check - post bottleneck', torch.max(x))

        # Global average pooling
        x = x.mean(dim=[2, 3, 4])  # Pool over spatial dimensions

        # Classification
        x = self.fc(x)
        print('Value check - post fc', x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device
