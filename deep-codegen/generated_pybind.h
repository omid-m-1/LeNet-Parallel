inline void export_kernel(py::module &m) { 
    m.def("linear_with_bias",[](py::capsule& input1, py::capsule& input2, py::capsule& output1, py::capsule& bias, bool is_forward, bool is_grad_bias){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
        array1d_t<float> bias_array = capsule_to_array1d(bias);
    return linear_with_bias(input1_array, input2_array, output1_array, bias_array, is_forward, is_grad_bias);
    }
  );
}