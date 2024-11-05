import tensorflow as tf
import gp_apis

def linear_with_bias(input1, input2, dim1_0, dim1_1, bias, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return linear_with_bias_real(X1, X2, dim1_0, dim1_1, bias, device0)
    return _lambda(input1, input2)

def linear_with_bias_real(input1, input2, dim1_0, dim1_1, bias, device0):
    out = gp_apis.gp_linear_with_bias(input1, input2, dim1_0, dim1_1, bias, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_linear_with_bias(dZ1, dZ2, dim1_0, dim1_1, bias, device0)
    return out, grad

