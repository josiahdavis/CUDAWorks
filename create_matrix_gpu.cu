#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
// Each thread populates one random matrix value
__global__ void initializeMatrix(float *matrix, int rows, int cols, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows){
        int index = idy * cols + idx;
        // Init random number generator
        curandState state;
        curand_init(seed, index, 0, &state);
        // Random number between -5, 5
        matrix[index] = -5.0f + 10.0f * curand_uniform(&state);
    }
}
void print_matrix(float *A, int M, int N){
    int i;
    printf("[");
    for (i = 0; i < M * N; i++){
        // End of row
        if ((i + 1)% N == 0){
            printf("%5.2f ", A[i]);
            if (i + 1 < M * N)
                printf("\n");
        }
        // Start of new row
        else if ((i + 1) % N == 1 && i > 0)
            printf("%6.2f, ", A[i]);
        // Within the given row
        else
            printf("%5.2f, ", A[i]);
    }
    printf("]\n");
}
int main(){
    int M = 3; // rows
    int N = 4; // cols
    float *h_matrix = new float[M * N];
    // Allocate memory on device
    // size_t is int with larger range since only positive. Commonly used for counting bytes/ memory functions.
    size_t size = M  * N * sizeof(float);
    float *d_matrix;
    cudaMalloc((void **)&d_matrix, size);
    // We will use 32 * 32 = 1024 threads per block
    // we do not need this many, but creating threads is cheap operation and we have implemented guard
    dim3 blockDim(32, 32, 1); 
    // We will use as many blocks as we need, in this case only 1
    dim3 gridDim(1, 1, 1);
    // Launch the kernel
    unsigned int seed = time(NULL);
    initializeMatrix<<<gridDim, blockDim>>>(d_matrix, M, N, seed);
    // Copy matrix to host
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    // Print from host
    print_matrix(h_matrix, M, N);
    // Free device memory
    cudaFree(d_matrix);
    // Free host memory
    delete[] h_matrix;
    return 0;
}