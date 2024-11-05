import torch as th
import torch.utils.dlpack
from . import graphpy as gpk # Import custom CUDA kernel
# Kernel API
def gp_linear_with_bias(input1, input2, bias, dim1_0, dim1_1, is_forward, is_grad_bias):
    # Convert tensors to DLPack format
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    bias_dl = th.utils.dlpack.to_dlpack(bias)
    # Initialize result tensor
    res1 = th.zeros(dim1_0, dim1_1, device = input1.device.type)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    # Run linear layer using Cuda Kernel
    gpk.linear_with_bias(input1_dl, input2_dl, res_dl1, bias_dl, is_forward, is_grad_bias)
    return res1
