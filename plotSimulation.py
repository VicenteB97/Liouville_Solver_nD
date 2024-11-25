import numpy as np
import matplotlib.pyplot as plt

def read_3d_tensor_from_binary(file_path, shape, dtype=np.float32):
    """
    Reads a binary file containing a 3D tensor.

    Args:
        file_path (str): Path to the binary file.
        shape (tuple): Shape of the 3D tensor (depth, height, width).
        dtype: Data type of the tensor elements (default: np.float32).

    Returns:
        np.ndarray: The 3D tensor.
    """
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape(shape)

def plot_slices_in_grid(tensor, num_levels=10, remove_lowest=True):
    """
    Plots all 2D slices of a 3D tensor in a single figure with subplots,
    allowing customization of contour levels and removing the lowest level.

    Args:
        tensor (np.ndarray): The 3D tensor.
        num_levels (int): Number of contour levels to display.
        remove_lowest (bool): Whether to remove the lowest contour level.
    """
    depth = tensor.shape[0]
    cols = int(np.ceil(np.sqrt(depth)))
    rows = int(np.ceil(depth / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i in range(depth):
        ax = axes[i]
        data = tensor[i]
        
        # Compute contour levels
        min_val, max_val = data.min(), data.max()
        levels = np.linspace(min_val, max_val, num_levels)
        if remove_lowest:
            levels = levels[1:]  # Remove the lowest level

        contour = ax.contour(data, levels=levels, cmap='viridis')
        fig.colorbar(contour, ax=ax, orientation='vertical')
        ax.set_title(f"Slice {i + 1}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

    # Hide unused subplots if depth < rows * cols
    for i in range(depth, rows * cols):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def plot_slices_sequentially(tensor, num_levels=10, pause_time_s:float = 1, remove_lowest=True):
    """
    Plots 2D slices of a 3D tensor sequentially in a single figure,
    allowing customization of contour levels and removing the lowest level.

    Args:
        tensor (np.ndarray): The 3D tensor.
        num_levels (int): Number of contour levels to display.
        remove_lowest (bool): Whether to remove the lowest contour level.
    """
    depth = tensor.shape[0]
    plt.figure()

    for i in range(depth):
        plt.clf()
        data = tensor[i]
        
        # Compute contour levels
        min_val, max_val = data.min(), data.max()
        levels = np.linspace(min_val, max_val, num_levels)
        if remove_lowest:
            levels = levels[1:]  # Remove the lowest level

        contour = plt.contour(data, levels=levels, cmap='viridis')
        plt.colorbar(contour)
        plt.title(f"Slice {i + 1}/{depth}")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.pause(0.3)  # Pause for .3 second between frames

    plt.show()


# Example usage
if __name__ == "__main__":
    # File path to the binary file
    file_path = "out/VDP_1_Mean_PDFs_0.bin"

    # Define the shape of the 3D tensor (depth, height, width)
    shape = (101, 1024, 1024)  # Example: 10 slices of 64x64

    # Read the tensor
    tensor = read_3d_tensor_from_binary(file_path, shape)

    # If you want to see fewer elements, you can use the plot_tensor
    # plot_tensor = tensor[::2]

    # Option 1: Display all slices in a grid
    # plot_slices_in_grid(plot_tensor)

    # Option 2: Display slices sequentially (uncomment to use)
    plot_slices_sequentially(tensor,num_levels=25, pause_time_s=0.1, remove_lowest=True)
