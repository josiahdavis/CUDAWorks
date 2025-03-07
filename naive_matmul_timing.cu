#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*                                                                                                        
   Naive computation of matmul: AxB = C             
   A is (m,k)                                       
   B is (k,n)               
   C is (m,n)

   Using x to index row of output, y to index col in output
*/                  

__global__ void naive_matmul(float* A, float* B, float* C, int M, int N, int K){
    // Each kernal call will compute one output element in c
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N){
        float dot = 0.0f;
        for (int k = 0; k < K; k++){
            // formula for indexing into a matrix to get correct element:
            //    row index * width + column index
            //    a: row index is i, width is K, col index is k
            //    b: row index is k, width is N, col index is j
            //    c: row index is i, width is N, col index is j
            dot += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = dot;
    }
}

__global__ void init_matrix(float*matrix, int M, int N, unsigned int seed){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N){
        int index = i * N + j;
        curandState state;
        curand_init(seed, index, 0, &state);
        matrix[index] = -2.0f + 4.0f * curand_uniform(&state);
    }
}

#define CEIL_DIV(X, Y) (X + Y - 1) / Y

int main(){

    // Setup timer
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create matrix on GPU
    int M = 4092;
    int K = 4092;
    int N = 4092;
    float *A;
    float *B;
    float *C;
    cudaMalloc((void **)&A, M * K * sizeof(float));
    cudaMalloc((void **)&B, K * N * sizeof(float));
    cudaMalloc((void **)&C, M * N * sizeof(float));
    dim3 blockDim(32, 32, 1);                             // how many threads in our block (e.g., 32 * 32 => 1024)
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);    // how many blocks in our grid (e.g., 4092/32 => 127.875 => 128)
    unsigned int seed = time(NULL);
    init_matrix<<<gridDim, blockDim>>>(A, M, K, seed);
    init_matrix<<<gridDim, blockDim>>>(B, K, N, seed+1);
    init_matrix<<<gridDim, blockDim>>>(C, M, N, seed+2);

    // Peform matmul
    cudaEventRecord(start);
    naive_matmul<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print summary
    // For each output C[i,j] we perform 2 * K operations, one multiply and one add
    // We do this for every row and column, hence FLOPs = 2 * K * M * N
    float seconds = elapsedTime / 1000.0f;
    double GFLOPs = (2.0 * M * N * K) / 1e9;                    
    double GFLOPs_s = (2.0 * M * N * K) / seconds / 1e9;
    printf("A [%d, %d] x B [%d, %d] = C [%d, %d]\n", M, K, K, N, M, N);
    printf("Kernel execution time: %6.4f ms\n", elapsedTime);
    printf("GFLOPs: %6.2f\n\n", GFLOPs);
    printf("GFLOPs/s: %6.2f\n\n", GFLOPs_s);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}

/*
Example output (on a A100):

A [4092, 4092] x B [4092, 4092] = C [4092, 4092]
Kernel execution time: 155.4565 ms
GFLOPs: 137.04

GFLOPs/s: 881.51
*/
