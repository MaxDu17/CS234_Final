import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    layer_list = []
    layer_list.append(nn.Linear(input_size, size))
    layer_list.append(nn.ReLU())
    for i in range(n_layers - 1):
        layer_list.append(nn.Linear(size, size))
        layer_list.append(nn.ReLU())
    layer_list.append(nn.Linear(size, output_size))
    model = nn.Sequential(*layer_list)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x

# build_mlp(10, 20, 3, 50)
