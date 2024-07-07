'''
CA: This is a re-creation of Neural Cellular Automata inspired by https://distill.pub/2020/growing-ca/

We start by doing a simple version where you can use a drawing app to create your own image for the 
network to 'grow'.

Use the xf env
'''

import configparser
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops

config = configparser.ConfigParser()
config.read('config.ini')

X_DIM = int(config['main']['X_DIM'])
Y_DIM = int(config['main']['Y_DIM'])
CELL_DIM = int(config['main']['CELL_DIM'])
HIDDEN_DIM = int(config['main']['HIDDEN_DIM'])
KERNEL_SIZE = int(config['main']['KERNEL_SIZE'])
N_CHAN = int(config['main']['N_CHAN'])

print(f'CELL_DIM = {CELL_DIM}')

def tensor_checks(tensor1, tensor2, threshold=1e-5, details=False, plot=False):
    '''
    Gives you some basic info on two tensors (intended for when you expect them to
    be the same but they aren't).
    '''
    print(f"===== Begin Tensor Checks =====")
    # Basic checks
    assert tensor1.shape == tensor2.shape
    assert tensor1.numel() == tensor2.numel()

    if torch.allclose(tensor1, tensor2, atol=threshold):
        print(f"\tTensors are equal within threshold {threshold}")
    else:
        # Summary stats
        print(f"\tShape: {tensor1.shape}")
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff)
        print(f"\tMaximum difference: {max_diff.item()}")
        print(f"\tMean difference: {torch.mean(diff).item()}")
        print(f"\tMedian difference: {torch.median(diff).item()}")
        print(f"\tStandard deviation of differences: {torch.std(diff).item()}")
        num_diff = torch.sum(diff > threshold).item()
        print(f"\tNumber of elements differing by more than {threshold}: {num_diff} of {tensor1.numel()} total elements")

        # Check for any NaNs / Inf
        print(f"\tNaN in tensor1: {torch.isnan(tensor1).any().item()}")
        print(f"\tNaN in tensor2: {torch.isnan(tensor2).any().item()}")
        print(f"\tInf in tensor1: {torch.isinf(tensor1).any().item()}")
        print(f"\tInf in tensor2: {torch.isinf(tensor2).any().item()}")

        # List individual differences if desired
        if details:
            max_diff_index = torch.argmax(diff)
            print(f"\tIndex of maximum difference: {max_diff_index.item()}")
            print(f"\tValue in tensor1: {tensor1.flatten()[max_diff_index].item()}")
            print(f"\tValue in tensor2: {tensor2.flatten()[max_diff_index].item()}")

            indices = torch.where(diff > threshold)
            print("\tIndices where difference > threshold:", indices)
            print("\tValues in tensor1:", tensor1[indices])
            print("\tValues in tensor2:", tensor2[indices])

        # Plot the difference if desired
        if plot:
            import matplotlib.pyplot as plt
    
            plt.imshow(diff.cpu().numpy())
            plt.colorbar()
            plt.title("Difference between tensor1 and tensor2")
            plt.show()

    print(f"=====  End Tensor Checks  =====")


class FixedDepthwiseConv2d(nn.Module):
    def __init__(self, kernel: torch.Tensor, stride: int=1, padding: int=0):
        super().__init__()
        
        assert kernel.dim() == 4, "Input kernel should contain 4 dimensions: cell_dim, 1, height, width"

        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.groups = kernel.shape[0] # for depthwise conv

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding, groups=self.groups)


class CA(nn.Module):
    def __init__(self, x_dim=X_DIM, y_dim=Y_DIM, cell_dim=CELL_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, device="cpu") -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.cell_dim = cell_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.device = device

        # print()
        # print("="*10 + " CA INIT " + "="*10)
        # print()
        # print(f'model device = {self.device}')
        # print()

        # Perception filters
        sobel_x = np.outer([1., 2., 1.], [-1., 0., 1.]) / 8.0 # Sobel
        sobel_y = sobel_x.T
        identity = np.float32([0., 1., 0.])
        identity = np.outer(identity, identity)

        # Add two leading dims, repeat first guy cell_dim times
        # All three need to end up as (cell_dim, 1, height, width)
        sobel_x = np.expand_dims(sobel_x, axis=(0, 1))
        sobel_x = np.repeat(sobel_x, self.cell_dim, axis=0)
        sobel_x = torch.tensor(sobel_x, dtype=torch.float32).to(self.device)
        sobel_y = np.expand_dims(sobel_y, axis=(0, 1))
        sobel_y = np.repeat(sobel_y, self.cell_dim, axis=0)
        sobel_y = torch.tensor(sobel_y, dtype=torch.float32).to(self.device)
        identity = np.expand_dims(identity, axis=(0, 1))
        identity = np.repeat(identity, self.cell_dim, axis=0)
        identity = torch.tensor(identity, dtype=torch.float32).to(self.device)

        # print(f'sobel_x shape: {sobel_x.shape}')
        # print(f'sobel_y shape: {sobel_y.shape}')
        # print(f'identity shape: {identity.shape}')
        # print()
        assert sobel_x.shape[0] == sobel_y.shape[0] == identity.shape[0] == self.cell_dim
        assert sobel_x.shape[1] == sobel_y.shape[1] == identity.shape[1] == 1
        assert sobel_x.shape[2] == sobel_y.shape[2] == identity.shape[2] == self.kernel_size
        assert sobel_x.shape[3] == sobel_y.shape[3] == identity.shape[3] == self.kernel_size

        self.grad_x_conv2d = FixedDepthwiseConv2d(sobel_x, padding=1)
        self.grad_y_conv2d = FixedDepthwiseConv2d(sobel_y, padding=1)
        self.identity_conv2d = FixedDepthwiseConv2d(identity, padding=1)

        self.layer1 = nn.Linear(3*self.cell_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_dim, self.cell_dim)

#        nn.init.constant_(self.layer1.weight, 0.0)  # Initialize weights
#        nn.init.constant_(self.layer1.bias, 0.0)  # Initialize biases
        nn.init.constant_(self.layer2.weight, 0.0)  # Initialize weights
#        nn.init.constant_(self.layer2.bias, 0.0)  # Initialize biases


    def perceive(self, x):
        grad_x = self.grad_x_conv2d(x)
        grad_y = self.grad_y_conv2d(x)
        cell_identity = self.identity_conv2d(x)

        # print("="*10 + " CA PERCEIVE " + "="*10)
        # print()
        # print(f'grad_x shape: {grad_x.shape}')
        # print(f'grad_y shape: {grad_y.shape}')
        # print(f'cell_identity shape: {cell_identity.shape}')
        # print()

        perception_grid = torch.concat((grad_x, grad_y, cell_identity), axis=1)

        # print(f'perception_grid shape: {perception_grid.shape}')
        # print()

        return perception_grid

    def extract_visible_grid(self, x):
        # We'll just use the first one in the batch
        return x[0, :4, :, :]

    def extract_visible_grid_batch(self, x):
        return x[:, :4, :, :]

    def get_alive_mask(self, x, threshold=0.1):
    
        max_pooled = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Create the mask by checking if the maximum value in the max-pooled tensor is greater than 0.1
        mask = (max_pooled > threshold).float()
    
        return mask

    def forward(self, x):

        # print("="*10 + " CA FWD " + "="*10)
        # print()

        perception_grid = self.perceive(x).to(self.device)

        # Flatten the H, W dims and rearrange
        perception_grid = einops.rearrange(perception_grid, 'b c h w -> b (h w) c')

        # print(f'perception_grid shape after flattening: {perception_grid.shape}')

        update_grid = self.layer1(perception_grid)
        update_grid = self.relu(update_grid)
        # print(f'update_grid shape before layer2: {update_grid.shape}')
        update_grid = self.layer2(update_grid)

        # print(f'update_grid shape after layers: {update_grid.shape}')

        # Rearrange back
        update_grid = einops.rearrange(update_grid, 'b (h w) c -> b c h w', h=self.x_dim, w=self.y_dim)

        # Stochastic masking of update grid before applying to state grid
        
        # stochastic_mask = torch.rand_like(update_grid[0, :, :, :]) # This does the same stochastic grid for all items in a batch
        stochastic_mask = torch.rand_like(update_grid) # Do unique stochastic grid for each item in a batch

        stochastic_mask = torch.where(stochastic_mask > 0.5, 1.0, 0.0)
        update_grid = update_grid * stochastic_mask

        # Alive masking of update grid before applying to state grid
        # (mask will be broadcast to the right size)
        alive_mask = self.get_alive_mask(x[0, 3, :, :].unsqueeze(0).unsqueeze(0))

        # Not clear to me which of these is really preferable <- I think all-grid is correct
        state_grid = (x + update_grid) * alive_mask # all-grid masking
#        state_grid = x + (update_grid * alive_mask) # update masking

        # print(f'state_grid shape after rearranging: {state_grid.shape}')

        # I messed with this a bit, thinking it might help, but it never seems to converge
        # state_grid = torch.sigmoid(state_grid)

#        visible_grid = self.extract_visible_grid(state_grid)
        visible_grid_batch = self.extract_visible_grid_batch(state_grid)

        # print(f'visible_grid shape: {visible_grid.shape}')

        return state_grid, visible_grid_batch


if __name__ == '__main__':

    print(f'See the nca notebook (.ipynb) instead')
