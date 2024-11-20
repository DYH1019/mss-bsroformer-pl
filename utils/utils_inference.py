import torch
from tqdm import tqdm
# from collections import OrderedDict
from typing import OrderedDict
from pathlib import Path
import typing as tp

"""
    When PyTorch Lightning saves a model, 
    it usually stores the model weights in state-dict and adds a 'model' to the name of each weight The prefix.

    By using the load_pl_state-dict function, 
    you can obtain a standard model weight dictionary without prefixes after loading, 
    which is convenient for loading into regular PyTorch models.
"""

def load_pl_state_dict(
        path: Path, device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """
    Loads and preprocesses pytorch-lightning state dict
    """

    sd = torch.load(path, map_location=device)

    new_sd = OrderedDict()

    for k, v in sd['state_dict'].items():
        
        """
        Only handle items with 'model' in their key names.
        This is usually because when PyTorch Lightning models are saved, 'model.' is added before the key names of all model weights The prefix.
        The purpose here is to remove 'model.' The prefix is used to adapt it to the standard PyTorch model.
        Move the 'model.' in the key Remove the prefix and keep the pure model layer name. The new_std obtained in this way is a cleaned dictionary suitable for loading ordinary PyTorch models.
        """

        if 'model' in k:
            new_sd[k.replace('model.', '')] = v
    return new_sd

