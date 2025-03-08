#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*                                                                                                        
   Matmul with global memory coalescing: AxB = C
   A is (m,k)                                       
   B is (k,n)               
   C is (m,n)

   Using x to index row of output, y to index col in output
   
   Ref:
     Simon Boehm's blog: https://siboehm.com/articles/22/CUDA-MMM
*/                  

void printMatrix(const float* matrix, int rows, int cols, const char* name){
    printf("Matrix %s showing top-left 5x10\n", name);
    for (int i = 0; i < 5 && i < rows; i++){
        for (int j = 0; j < 10 && j < cols; j++){
            printf("%6.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template <const uint BLOCK_SIZE>
__global__ void coalesceMatmul(float* A, float* B, float* C, int M, int N, int K){
    // Each kernal call will compute one output element in c
    // Previously what we had:
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    int j = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    if (i < M && j < N){
        // printf("(%d, %d) blockDim.x = %d BLOCK_SIZE=%d\n", i, j, blockDim.x, BLOCK_SIZE);
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


void setSeed(){
    srand(time(NULL));
}

void initMatrix(float *matrix, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            float random_value = (float)rand() / RAND_MAX;
            matrix[i * N + j] = random_value * 4.0f - 2.0f;
            // matrix[i * N + j] = (float)(rand() % 11); // to get whole numbers
        }
    }
}

#define CEIL_DIV(X, Y) (X + Y - 1) / Y

int main(){

    // Setup timer
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create matrix on CPU
    int M = 4092;
    int K = 4092;
    int N = 4092;
    float *A_h = (float *)malloc(M * K * sizeof(float));
    float *B_h = (float *)malloc(K * N * sizeof(float));
    float *C_h = (float *)malloc(M * N * sizeof(float));
    setSeed();
    initMatrix(A_h, M, K);
    initMatrix(B_h, K, N);
    
    // Transfer matrices to GPU
    float *A;
    float *B;
    float *C;
    cudaMalloc((void **)&A, M * K * sizeof(float));
    cudaMalloc((void **)&B, K * N * sizeof(float));
    cudaMalloc((void **)&C, M * N * sizeof(float));
    cudaMemcpy(A, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Peform matmul
    dim3 blockDim(32 * 32);                              // how many threads in each block (e.g., 32 * 32 => 1024)
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));      // how many blocks in our grid (e.g., 4092/32 => 127.875 => 128)
    cudaEventRecord(start);
    coalesceMatmul<32><<<gridDim, blockDim>>>(A, B, C, M, N, K); // Note BLOCKSIZE=32 value inserted
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print summary
    printf("A [%d, %d] x B [%d, %d] = C [%d, %d]\n", M, K, K, N, M, N);
    float seconds = elapsedTime / 1000.0f;
    printf("Kernel execution time: %6.4f ms\n", elapsedTime);

    // For each output C[i,j] we perform 2 * K operations, one multiply and one add
    // We do this for every row and column, hence FLOPs = 2 * K * M * N
    double GFLOPs = (2.0 * M * N * K) / 1e9;
    double GFLOPs_s = (2.0 * M * N * K) / seconds / 1e9;
    printf("GFLOPs: %6.2f\n\n", GFLOPs);
    printf("GFLOPs/s: %6.2f\n\n", GFLOPs_s);

    // Memory operations are reading and writing 
    // Min reading needed: matrix A (M*K), matrix B (K*N), and matrix C (M*N)
    // Min writing needed: Matrix C (M*N)
    // Using binary prefix instead of decimal prefix
    // e.g., 1024 bytes in a KiB, 1024^2 bytes in MiB, 1024^3 bytes in GiB
    float total_bytes = (M*K + K*N + M*N*2) * sizeof(float);
    float throughput_gb_s = (total_bytes / seconds) / (1024 * 1024 * 1024);
    printf("Memory throughput: %.2f GB/s\n", throughput_gb_s);

    // Print output
    cudaMemcpy(C_h, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(A_h, M, K, "A");
    printMatrix(B_h, K, N, "B");
    printMatrix(C_h, M, N, "C");
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}

/*
Example output (on a A100):

A [4092, 4092] x B [4092, 4092] = C [4092, 4092]
Kernel execution time: 48.4587 ms
GFLOPs: 137.04

GFLOPs/s: 2827.91

Memory throughput: 5.15 GB/s
*/
