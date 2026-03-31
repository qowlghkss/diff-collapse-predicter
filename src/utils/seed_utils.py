import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across all relevant modules.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Global seed set to: {seed}")
