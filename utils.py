"""Some code shared across files."""

import torch


def get_device():
    """Get the available device."""
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
