import random

import numpy as np
import torch


def set_seed(seed: int) -> int:
    """Seed every RNG we use. Call at the top of every script and log the returned seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
