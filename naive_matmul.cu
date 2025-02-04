#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
/*                                                                                                        
   Naive computation of matmul: AxB = C             
   A is (m,k)                                       
   B is (k,n)               
   C is (m,n)

   Using x to index row, y to index col in output                     
*/                  

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
            printf("%7.3f, ", A[i]);
        // Within the given row
        else
            printf("%5.2f, ", A[i]);
    }
    printf("]\n");
}

__global__ void initializeMatrix(float*matrix, int m, int n, unsigned int seed){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < n){
        int index = x * n + y;
        curandState state;
        curand_init(seed, index, 0, &state);
        matrix[index] = -2.0f + 4.0f * curand_uniform(&state);
    }
}
__global__ void matmul(float* a, float* b, float* c, int m, int n, int k){
    // Each kernal call will compute one output element in c
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < n){
        float dot = 0.0f;
        for (int i = 0; i < k; i++){
            // formula for indexing into a matrix to get correct element:
            //    row index * width + column index
            //    a: row index is x, width is k, col index is i
            //    b: row index is i, width is n, col index is y
            //    c: row index is x, width is n, col index is y
            dot += a[x * k + i] * b[i * n + y];
        }
        c[x * n + y] = dot;
    }
}
int main(){

    // Create matrix on GPU
    int M = 2;
    int K = 3;
    int N = 4;
    float *A;
    float *B;
    float *C;
    cudaMalloc((void **)&A, M * K * sizeof(float));
    cudaMalloc((void **)&B, K * N * sizeof(float));
    cudaMalloc((void **)&C, M * N * sizeof(float));
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(1, 1, 1);
    unsigned int seed = time(NULL);
    initializeMatrix<<<gridDim, blockDim>>>(A, M, K, seed);
    initializeMatrix<<<gridDim, blockDim>>>(B, K, N, seed+1);
    initializeMatrix<<<gridDim, blockDim>>>(C, M, N, seed+2);

    // Peform matmul
    matmul<<<gridDim, blockDim>>>(A, B, C, M, N, K);

    // Print output
    float *A_h = new float[M * K];
    float *B_h = new float[K * N];
    float *C_h = new float[M * N];
    cudaMemcpy(A_h, A, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(A_h, M, K);
    print_matrix(B_h, K, N);
    print_matrix(C_h, M, N);

    // Clean up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}
