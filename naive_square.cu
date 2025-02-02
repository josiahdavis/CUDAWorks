/*                                                                                                        
   Naive computation of matmul: AxB = C             
   A is (n,n)                                       
   B is (n,n)               
   C is (n,n)                     
*/                        
#include <stdio.h>                          
#include <curand_kernel.h> 
#include <cuda_runtime.h>
__global__ void initializeMatrix(float*matrix, int n, unsigned int seed){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < n){
        int index = row * n + col;
        curandState state;
        curand_init(seed, index, 0, &state);
        matrix[index] = -2.0f + 4.0f * curand_uniform(&state);
    }
}
__global__ void matmul(int n, float* a, float* b, float* c){
    // Each kernal call will compute one output element in c
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n){
        float dot = 0.0f;
        for (int i = 0; i < n; i++){
            // formula for indexing into a matrix to get correct element:
            //    row index * width + column index
            //    a: row index is row, col index is i
            //    b: row index is i, col index is col
            //    c: row index is row, col index is col
            dot += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = dot;
    }
}
int main(){
    int N = 5;
    float *A;
    float *B;
    float *C;
    cudaMalloc((void **)&A, N * N * sizeof(float));
    cudaMalloc((void **)&B, N * N * sizeof(float));
    cudaMalloc((void **)&C, N * N * sizeof(float));
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(1, 1, 1);
    unsigned int seed = time(NULL);
    initializeMatrix<<<gridDim, blockDim>>>(A, N, seed);
    initializeMatrix<<<gridDim, blockDim>>>(B, N, seed);
    initializeMatrix<<<gridDim, blockDim>>>(C, N, seed);
    matmul<<<gridDim, blockDim>>>(N, A, B, C);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
