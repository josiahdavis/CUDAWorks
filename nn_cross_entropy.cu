#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_matrix(float* mat, int rows, int cols){
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i < rows && col_j < cols){
        int index = row_i * cols + col_j;
        curandState state;
        curand_init(123, index, 0, &state);
        mat[index] = curand_normal(&state)*sqrtf(2.f/rows);
    }
}

__global__ void init_zero_one_uniform(float* mat, int rows, int cols){
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i < rows && col_j < cols){
        int index = row_i * cols + col_j;
        curandState state;
        curand_init(123, index, 0, &state);
        mat[index] = curand_uniform(&state);
    }
}

__global__ void cross_entropy(float* preds, float* actual, float* output, int batch_size, int class_size){
    // preds.shape =  (batch_size, clas_size)
    // actual.shape = (batch_size, class_size)
    // output.shape = (batch_size,)
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size){
        float loss = 0.f;
        for (int j = 0; j < class_size; j++){
            loss -= actual[batch_idx*class_size + j] * log(max(1e-6, preds[batch_idx*class_size + j]));
        }
        output[batch_idx] = loss;
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* name){
    printf("Matrix %s showing top-left 5x10\n", name);
    for (int i = 0; i < 5 && i < rows; i++){
        for (int j = 0; j < 10 && j < cols; j++){
            printf("%5.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(){
    int BLOCK_SIZE = 16;
    
    // Set up parameters
    int batch_size = 4;
    int n_classes = 6;
    
    // Allocate memory for GPU
    float *d_preds;
    float *d_actuals;
    float *d_losses;
    cudaMalloc((void **)&d_preds, batch_size * n_classes * sizeof(float));
    cudaMalloc((void **)&d_actuals, batch_size * n_classes * sizeof(float));
    cudaMalloc((void **)&d_losses, batch_size * sizeof(float));

    // Initialize weights GPU
    dim3 dimGrid = dim3(ceil(batch_size/(float)BLOCK_SIZE), ceil(n_classes/(float)BLOCK_SIZE), 1);
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    // Actual labels will be 0 or 1
    init_matrix<<<dimGrid, dimBlock>>>(d_actuals, batch_size, n_classes);
    
    // NN outputs will be probabilities from 0-1
    init_zero_one_uniform<<<dimGrid, dimBlock>>>(d_preds, batch_size, n_classes);

    // Calculate cross entropy losses
    cross_entropy<<<dimGrid, dimBlock>>>(d_preds, d_actuals, d_losses, batch_size, n_classes);
    
    // Copy to CPU
    float *h_preds = new float[batch_size * n_classes];
    float *h_actuals = new float[batch_size * n_classes];
    float *h_losses = new float[batch_size];
    cudaMemcpy(h_preds, d_preds, batch_size * n_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_actuals, d_actuals, batch_size * n_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_losses, d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Inspect outputs
    print_matrix(h_preds, batch_size, n_classes, "Preds");
    print_matrix(h_actuals, batch_size, n_classes, "Actuals");
    print_matrix(h_losses, batch_size, 1, "Losses");

    // Calculate total loss
    float cumulative_loss = 0.f;
    for (int b = 0; b < batch_size; b++){
        cumulative_loss += h_losses[b];
    }
    printf("Total loss %6.4f\n", cumulative_loss);
    return 0;
}
