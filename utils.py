import subprocess
import torch

# Play a sound so I know when stuff is done
def alert():
    subprocess.run(["aplay", "/usr/lib/libreoffice/share/gallery/sounds/apert.wav"])

# A version of init_grid that takes in lists with x_inits, y_inits for starting pixels
def init_grid_specified_inits(x_inits, y_inits, batch_size=1, cell_dim=16, x_dim=28, y_dim=28):
    state_grid = torch.zeros((batch_size, cell_dim, x_dim, y_dim), dtype=torch.float32)
    for i in range(len(x_inits)):
        state_grid[i, 3, x_inits[i], y_inits[i]] = 1.
    
    return state_grid


# Function to find the min and max values for each channel
# Assumes a tensor in, with the RGB info in the first dim
def find_min_max(tensor_image: torch.Tensor) -> None:
    min_values = tensor_image.view(4, -1).min(dim=1).values
    max_values = tensor_image.view(4, -1).max(dim=1).values

    print("Min values for each channel:", min_values)
    print("Max values for each channel:", max_values)
