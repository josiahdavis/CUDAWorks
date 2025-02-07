#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_matrix(float* mat, int rows, int cols){
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i < rows && col_j < cols){
        int index = row_i * rows + col_j;
        curandState state;
        curand_init(123, index, 0, &state);
        mat[index] = curand_normal(&state)*sqrtf(2.f/rows);
    }
}

__global__ void relu(float* mat_in, float* mat_out, int rows, int cols){
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i < rows && col_j < cols){
        int index = row_i * rows + col_j;
        float activation = mat_in[index];
        mat_out[index] = activation > 0.f ? activation : 0.f;
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* name){
    printf("Matrix %s showing top-left 5x10\n", name);
    for (int i = 0; i < 5 && i < rows; i++){
        for (int j = 0; j < 10 && j < cols; j++){
            printf("%5.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(){
    int BLOCK_SIZE = 16;
    
    // Set up parameters
    int batch_size = 4;
    int n_features = 6;
    
    // Allocate memory for GPU
    float *d_in;
    float *d_out;
    cudaMalloc((void **)&d_in, batch_size * n_features * sizeof(float));
    cudaMalloc((void **)&d_out, batch_size * n_features * sizeof(float));

    // Initialize weights GPU
    dim3 dimGrid = dim3(ceil(batch_size/(float)BLOCK_SIZE), ceil(n_features/(float)BLOCK_SIZE), 1);
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    init_matrix<<<dimGrid, dimBlock>>>(d_in, batch_size, n_features);
    relu<<<dimGrid, dimBlock>>>(d_in, d_out, batch_size, n_features);
    
    // Copy to CPU
    float *h_in = new float[batch_size * n_features];
    float *h_out = new float[batch_size * n_features];
    cudaMemcpy(h_in, d_in, batch_size * n_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, batch_size * n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Inspect
    print_matrix(h_in, batch_size, n_features, "Input");
    print_matrix(h_out, batch_size, n_features, "Output");
    return 0;
}
