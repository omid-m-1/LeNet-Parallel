#pragma once
#include "csr.h"
#include "op.h"

void linear_with_bias(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array1d_t<float>& bias, bool is_forward, bool is_grad_bias);
