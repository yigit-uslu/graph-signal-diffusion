import os
import random
import numpy as np
import torch

def seed_everything(seed):
    # set the random seed
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def assert_same_dtype_and_size(tensor1, tensor2):
    # Check if both tensors have the same data type
    assert tensor1.dtype == tensor2.dtype, f"Data types do not match: {tensor1.dtype} vs {tensor2.dtype}"
    
    # Check if both tensors have the same size
    assert tensor1.size() == tensor2.size(), f"Sizes do not match: {tensor1.size()} vs {tensor2.size()}"


# def find_substring_index(string, substring):
#     return string.index(substring) + len(substring) - 1