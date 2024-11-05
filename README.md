# LeNet300 with Data parallelization - Assignment 4

## Usage

To train LeNet300 model on 2/1 GPUs run: `python main.py --kernel Kernel --max_rank 2/1` command. results for Custom CUDA kernel are saved in results folder.
 
For compiling the kernel, enter the following command in the `deep-codegen` directory:
```bash
mkdir build && cd build
cmake ..
make -j
cp graphpy.cpython-38-x86_64-linux-gnu.so ../
```
