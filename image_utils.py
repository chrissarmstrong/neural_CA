import torch
import numpy as np
import matplotlib.pyplot as plt


# Function to plot the grid (with optional alpha channel)
# Assume tensor input
def plot_CA(tensor_image: torch.Tensor, size=(5, 5)) -> None:
    plt.figure(figsize=size)  # Set the figure size

    nparr = np.transpose(tensor_image.detach().numpy(), (1, 2, 0))

    # Check if the input has an alpha channel
    if nparr.ndim == 3 and nparr.shape[2] == 4:
        # Split the RGBA image into RGB and alpha
        rgb = nparr[:, :, :3]
        alpha = nparr[:, :, 3]
        
        # Create a checkerboard pattern as the background
        checkerboard = create_checkerboard_pattern(rgb.shape[0], rgb.shape[1], square_size=1)
        
        # Combine the RGB image with the checkerboard pattern using alpha blending
        composite = rgb * alpha[..., None] + checkerboard * (1 - alpha[..., None])
        
        # Display the composite image
        plt.imshow(composite)
    else:
        # If no alpha channel, just display the image as before
        plt.imshow(nparr)
    
    plt.axis('off')  # Turn off axis
    plt.show()


# Function to create checkerboard background for alpha display
def create_checkerboard_pattern(height, width, square_size=10):
    # Create a checkerboard pattern
    pattern = np.zeros((height, width))
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                pattern[i:i+square_size, j:j+square_size] = 1
    
    # Scale the pattern to be between 0 and 1 and convert it to RGB
    pattern = np.repeat(pattern[:, :, np.newaxis], 3, axis=2)
    pattern = (pattern * 0.2 + 0.5)  # Adjust the brightness and contrast
    return pattern


# Function to plot two grids side by side (with optional alpha channel)
# Assume tensor input
def plot_two_CAs(tensor_image_1: torch.Tensor, tensor_image_2: torch.Tensor, size=(2, 1)) -> None:
    fig, axes = plt.subplots(1, 2, figsize=size)
    
    images = [tensor_image_1, tensor_image_2]
    
    for i, img in enumerate(images):
        nparr = img.detach().cpu().numpy()
        if nparr.ndim == 4 and nparr.shape[0] == 1:  # Handle single-image batch
            nparr = nparr.squeeze(0)
        nparr = np.transpose(nparr, (1, 2, 0))
        
        # Check if the input has an alpha channel
        if nparr.ndim == 3 and nparr.shape[2] == 4:
            # Split the RGBA image into RGB and alpha
            rgb = nparr[:, :, :3]
            alpha = nparr[:, :, 3]
            
            # Create a checkerboard pattern as the background
            checkerboard = create_checkerboard_pattern(rgb.shape[0], rgb.shape[1], square_size=1)
            
            # Combine the RGB image with the checkerboard pattern using alpha blending
            composite = rgb * alpha[..., None] + checkerboard * (1 - alpha[..., None])
            
            # Display the composite image
            axes[i].axis('off')  # Turn off axis
            axes[i].imshow(composite)
        else:
            # If no alpha channel, just display the image as before
            axes[i].axis('off')  # Turn off axis
            axes[i].imshow(nparr)

    plt.tight_layout()
    plt.show()



# Function to plot a grid of tensor images (with optional alpha channel)
def plot_tensors(tensor_list, figsize, row_elements):
    # Calculate the number of rows needed to display the tensors
    num_tensors = len(tensor_list)
    num_rows = (num_tensors + row_elements - 1) // row_elements
    
    # Create a figure with the specified size
    fig, axes = plt.subplots(num_rows, row_elements, figsize=figsize)
    
    if num_rows == 1:  # If there's only one row, axes is a 1D array
        axes = [axes]
    
    for i, tensor in enumerate(tensor_list):
        # Convert PyTorch tensor to numpy array and normalize to [0, 1]
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:  # RGB or RGBA
            # Convert from (C, H, W) to (H, W, C)
            img = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            raise ValueError("Unsupported tensor shape: {}".format(tensor.shape))
        
        # Normalize if needed
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        
        # Drop the alpha channel if present
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Get the current axis
        ax = axes[i // row_elements][i % row_elements]
        
        # Display the image
        ax.imshow(img)
        ax.axis('off')  # Turn off the axis
    
    # Hide any unused subplots
    for j in range(i + 1, num_rows * row_elements):
        axes[j // row_elements][j % row_elements].axis('off')

    plt.tight_layout()
    plt.show()


def plot_layers(tensor, size=(20, 5)):
    """
    Plots each layer of a 4-channel image tensor (RGB + alpha) separately.
    
    Parameters:
    - tensor: A PyTorch tensor of shape (4, 28, 28).
    """
    if tensor.shape[0] != 4 or len(tensor.shape) != 3:
        raise ValueError("Input tensor must be of shape (4, 28, 28)")
    
    # Clip the tensor values to the [0, 1] range to avoid clipping warnings
    tensor = torch.clamp(tensor, min=0, max=1)
    
    # Setting up the figure and axes for 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=size)
    
    # Titles for each subplot
    titles = ['R', 'G', 'B', r"$\alpha$"]
    
    for i, ax in enumerate(axes):
        # Selecting the channel
        channel = tensor[i]
        
        # For RGB channels, we create an empty image of zeros
        # and only modify the current channel to be displayed.
        if i < 3: # For RGB channels
            img = torch.zeros_like(tensor[:3])  # Creating a black image with all layers
            img[i] = channel  # Assigning the current channel's values
            ax.imshow(img.permute(1, 2, 0).numpy())  # Permuting dimensions to (28, 28, 3)
        else:  # For the alpha channel, display it in grayscale
            ax.imshow(channel.numpy(), cmap='gray')
        
        ax.set_title(titles[i])
        ax.axis('off')  # Turn off axis
    
    plt.tight_layout()
    plt.show()


def plot_combined(target_tensor, tensor, size=(25, 5)):
    """
    Plots target_tensor, then tensor, then layers of tensor
    Assumes both are tensors, and both have alpha channel
    """

    if target_tensor.shape[0] != 4 or len(target_tensor.shape) != 3:
        raise ValueError("Input target tensor must be of shape (4, 28, 28)")
    
    if tensor.shape[0] != 4 or len(tensor.shape) != 3:
        raise ValueError("Input tensor must be of shape (4, 28, 28)")
    
    # Clip the tensor values to the [0, 1] range to avoid clipping warnings
    target_tensor = torch.clamp(target_tensor, min=0, max=1)
    tensor = torch.clamp(tensor, min=0, max=1)
    
    # Setting up the figure and axes
    fig, axes = plt.subplots(1, 6, figsize=size)
    
    # Titles for each subplot
    titles = ['Target', 'Output', 'R', 'G', 'B', r"$\alpha$"]

    # First the target tensor and output tensor
    for i, item in enumerate([target_tensor, tensor, tensor[0], tensor[1], tensor[2], tensor[3]]):
        if i < 2:  # For target and output tensors
            nparr = item.detach().cpu().numpy()
            if nparr.ndim == 4 and nparr.shape[0] == 1:  # Handle single-image batch
                nparr = nparr.squeeze(0)
            nparr = np.transpose(nparr, (1, 2, 0))
            
            # Split the RGBA image into RGB and alpha
            rgb = nparr[:, :, :3]
            alpha = nparr[:, :, 3]
            
            # Create a checkerboard pattern as the background
            checkerboard = create_checkerboard_pattern(rgb.shape[0], rgb.shape[1], square_size=1)
            
            # Combine the RGB image with the checkerboard pattern using alpha blending
            composite = rgb * alpha[..., None] + checkerboard * (1 - alpha[..., None])
            
            # Display the composite image
            axes[i].set_title(titles[i])
            axes[i].axis('off')  # Turn off axis
            axes[i].imshow(composite)
        else:
            # Now the output tensor by layer
            tensor_ind = i - 2
            channel = item
        
            # For RGB channels, we create an empty image of zeros
            # and only modify the current channel to be displayed.
            if tensor_ind < 3: # For RGB channels
                img = torch.zeros_like(tensor[:3])  # Creating a black image with all layers
                img[tensor_ind] = channel  # Assigning the current channel's values
                axes[i].imshow(img.permute(1, 2, 0).numpy())  # Permuting dimensions to (28, 28, 3)
            else:  # For the alpha channel, display it in grayscale
                axes[i].imshow(channel.numpy(), cmap='gray')
            
            axes[i].set_title(titles[i])
            axes[i].axis('off')  # Turn off axis
    
    plt.tight_layout()
    plt.show()


def plot_losses(losses, ymax=None, size=(10, 6), title="Loss"):
    plt.figure(figsize=size)
    shifted_indices = [i + 1 for i in range(len(losses))]
    plt.plot(shifted_indices, losses)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Loss Value')
    if ymax is not None:
        plt.ylim(top=ymax)
    plt.ylim(bottom=0)
    plt.grid(True)  # Adds a grid for easier reading
    plt.show()


# Convert numpy to torch tensor
def np_to_tens(numpy_image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(numpy_image).permute(2, 0, 1).float() / 255.0


# Convert BW image to RGB
def bw_to_rgb(tensor_image: torch.Tensor) -> torch.Tensor:
    return tensor_image.repeat(3, 1, 1)


# Function to add an all-ones alpha channel to an RGB tensor image
def add_alpha_channel(tensor_image: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor_image, torch.Tensor):
        if tensor_image.shape[0] == 4:
            return tensor_image
        alpha_channel = torch.ones_like(tensor_image[0:1, :, :])
        return torch.cat((tensor_image, alpha_channel), dim=0)
    else:
        raise TypeError("Input must be a torch.Tensor")



# Function to apply the alpha mask to the RGB channels of a tensor
# This is kind of a special case arising from our NCA use case, where
# we want the alpha channel to mask out the RGB channels on the target
def apply_alpha_mask(tensor_image: torch.Tensor) -> torch.Tensor:
    masked_tensor = torch.zeros_like(tensor_image)
    for layer in range(3):
        masked_tensor[layer, :, :] = tensor_image[layer, :, :] * tensor_image[3, :, :]
    masked_tensor[3, :, :] = tensor_image[3, :, :]
    return masked_tensor


# Function to normalize a tensor image to the range [0, 1] (assumes a
# torch tensor, not numpy array) (do this before converting to RGB)
def normalize_bw_tensor_image(tensor_image):
    tensor_min = tensor_image.min()
    tensor_max = tensor_image.max()
    return (tensor_image - tensor_min) / (tensor_max - tensor_min)


# Image properties for troubleshooting
def print_img_props(img, comment="Image properties:"):
    print()
    print(comment)
    print(f"Image type: {type(img)}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image shape: {img.shape}")
    print(f"Image min, max: {img.min()}, {img.max()}")
    print(f"Image mean: {img.mean():.3f}")
    