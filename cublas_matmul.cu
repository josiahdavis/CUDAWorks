#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void printMatrix(const float* matrix, int rows, int cols, const char* name){
    printf("Matrix %s showing top-left 5x5", name);
    for (int i = 0; i < 5 && i < rows; i++){
        for (int j = 0; j < 5 && j < cols; j++){
            printf("%5.2d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrixMultiplyCublas(const int M, const int N, const int K){
    // Allocate host memory
    // M = the number of rows in C and A
    // N = the number of cols in C and B
    // K = the inner dimension, the columns in A and rows in B
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize matrices
    for (int i = 0; i < M * K; i++){
        h_A[i] = i;
        // h_A[i] = (float)rand()/(float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++){
        h_B[i] = i * 2;
        // h_B[i] = (float)rand()/(float)RAND_MAX;
    }
    
    // Print input matrices
    printMatrix(h_A, M, K, "A");
    printMatrix(h_B, K, N, "B");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle is required to use cuBLAS. It initializes the library 
    // and acts as a context for making cuBLAS function calls.
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f; // C = α*(A@B)+β*C, not sure what the purpose of beta param
    // Compute C = A x B
    // CUBLAS_OP_N: is the non-transpose operation is selected
    // M, N: The number of rows and cols in C
    // K: The inner dimension of A and B
    // leading dimension (lda, ldb, ldc) is not necessarily the number of rows—it's the stride (or pitch) between consecutive columns).
    // cuBLAS stores matrices in column-major order, meaning that elements of each column are stored contiguously in memory.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M); 

    // Copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print output matrix
    printMatrix(h_C, M, N, "C");

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    int M = 2;
    int K = 3;
    int N = 4;
    matrixMultiplyCublas(M, N, K);
    printf("Matrix multiplication completed successfully");
    return 0;
}