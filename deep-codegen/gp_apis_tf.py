import tensorlow as tf
import kernel as gpk
def gp_linear_with_bias(X1, X2, dim1_0, dim1_1, bias):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    X2_dl = tf.experimental.dlpack.to_dlpack(X2)
    bias_dl = tf.experimental.dlpack.to_dlpack(bias)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.linear_with_bias(X1_dl, X2_dl, res_dl, bias_dl)
    return res
