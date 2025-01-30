#include <stdio.h>

__global__ void helloFromGPU(){
    // Part 1: print each thread
    printf("Hello world from GPU block %d (out ot %d) and thread %d (out of %d)\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);

    // Part 2: print one time
    // Create a unique ID from grid, block, thread heirarchy
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x == 0) {
        printf("--- Kernel info --");

        // Grid dim tells you number of blocks
        printf("Grid dim = %d, %d, %d\n", gridDim.x, gridDim.y, gridDim.z);

        // Block dim tells you number of threads
        printf("Block dim = %d, %d, %d\n", blockDim.x, blockDim.y, blockDim.z);
    }
}

int main() {
    printf("Hello world, from CPU!\n");

    // Launch the kernel with 2 blocks and 8 threads
    helloFromGPU<<<2,8>>>();

    // Wait for GPU to finish before accessing results on CPU
    cudaDeviceSynchronize();

    return 0;
}