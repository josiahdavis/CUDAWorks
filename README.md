# CUDAWorks

Short stand-alone examples of how to use CUDA

1. [`hello_world.cu`](hello_world.cu): Program shows how to print from each thread, access threads, blocks and print kernel summary.
2. [`create_matrix_gpu.cu`](create_matrix_gpu.cu): Initialize random matrix on the GPU, copy to CPU and print.
3. [`vector_add.cu`](vector_add.cu): Initialize two random vectors on the CPU, copy to GPU and add together.
4. [`naive_square.cu`](naive_square.cu): Naive matmul with two square matrices.
5. [`naive_matmul.cu`](naive_matmul.cu): Naive matmul with two non-square matrices.
6. ['cublas_square.cu][cublas_square.cu]: Matrix multiplication using cublas (square matrices).

### Instructions

Run examples like so:
```
nvcc hello_world.cu -o main && ./main
```

Run cuBLAS examples like so:
```
nvcc cublas_square.cu -o main -lcublas && ./main
```