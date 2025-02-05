#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void printMatrix(const float* matrix, int N, const char* name){
    std::cout <<"Matrix " << name << " showing top-left 5x5): " << std::endl;
    for (int i = 0; i < 5 && i < N; i++){
        for (int j = 0; j < 5 && j < N; j++){
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void matrixMultiplyCublas(const int N){
    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++){
        h_A[i] = (float)rand()/(float)RAND_MAX;
        h_B[i] = (float)rand()/(float)RAND_MAX;
        // h_A[i] = static_cast<float>(rand() / RAND_MAX) * 10.0f;
        // h_B[i] = static_cast<float>(rand() / RAND_MAX) * 10.0f;
    }
    printf("%6.2f, %6.2f, %6.2f ", h_A[0], h_A[1], h_A[2]);
    printf("%6.2f, %6.2f, %6.2f ", h_B[0], h_B[1], h_B[2]);
    std::cout << "h_A[0]: " << h_A[0] << " h_B[0]: " << h_B[0] << std::endl;

    // Print input matrices
    printMatrix(h_A, N, "A");
    printMatrix(h_B, N, "B");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle is required to use cuBLAS. It initializes the library 
    // and acts as a context for making cuBLAS function calls.
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print output matrix
    printMatrix(h_C, N, "C");

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
    int N = 512;
    matrixMultiplyCublas(N);
    std::cout << "Matrix multiplication completed successfully" << std::endl;
    return 0;
}