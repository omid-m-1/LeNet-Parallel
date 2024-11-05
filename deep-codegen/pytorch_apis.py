import torch as th
from . import gp_apis

# Custom linear function
class custom_linear_with_bias_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, bias):
        # Output dimensions
        dim1_0 = input1.shape[0]
        dim1_1 = input2.shape[0]
        # Load custom kernel API
        res = gp_apis.gp_linear_with_bias(input1, input2, bias, dim1_0, dim1_1, True, False) # is_forward=True, bias=False
        # Cache inputs for backward step
        ctx.backward_cache = (input1, input2, bias)
        return res

    @staticmethod
    def backward(ctx, dZ): 
        # Load cached variables
        input1, input2, bias = ctx.backward_cache
        # Make transposed arrays contiguous in memory
        input1 = input1.t().contiguous()
        input2 = input2.t().contiguous()
        # Load custom kernel API
        grad_input = gp_apis.gp_linear_with_bias(dZ, input2, bias, dZ.shape[0], input2.shape[0], False, False) #is_forward=False, bias=False
        grad_weight = gp_apis.gp_linear_with_bias(dZ.t().contiguous(), input1, bias, dZ.shape[1], input1.shape[0], False, False) #is_forward=False, bias=False
        # compute bias gredient
        grad_bias = dZ.sum(0)
        return grad_input, grad_weight, grad_bias

# Apply custom linear function
def custom_linear_with_bias(input1, input2, bias):
    return custom_linear_with_bias_impl.apply(input1, input2, bias)
