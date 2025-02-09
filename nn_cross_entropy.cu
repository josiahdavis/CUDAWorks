#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_matrix_normal(float* mat, int rows, int cols){
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i < rows && col_j < cols){
        int index = row_i * cols + col_j;
        curandState state;
        curand_init(123, index, 0, &state);
        mat[index] = curand_normal(&state)*sqrtf(2.f/rows);
    }
}

__global__ void init_matrix_one_hot(float* matrix, int indices, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows){
        for (int j = 0; j < cols; ++j){
            matrix[i * cols + j] = (j == indices[i]) ? 1.f : 0.f;
        }
    }
}

__global__ void init_vector(int *output, int rows, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows){
        curandState state;
        curand_init(123, i, 0, &state);
        output[i] = (int)(curand_uniform(&state) * n);
    }
}

__global__ void softmax(float* mat_in, float* mat_out, int rows, int cols){
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_i < rows && col_j < cols){
        float maxval = mat_in[row_i * cols];
        for (int i = 1; i < cols; i ++){
            maxval = max(maxval, mat_in[row_i * cols + i]);
        }
        float divisor = 0.f;
        for (int i = 0; i < cols; i++){
            divisor += exp(mat_in[row_i*cols + i] - maxval);
        }
        mat_out[row_i * cols + col_j] = exp(mat_in[row_i * cols + col_j] - maxval) / divisor;
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
    float *d_logits;
    float *d_preds;
    float *d_actuals_idx;
    float *d_actuals;
    float *d_losses;
    cudaMalloc((void **)&d_logits, batch_size * n_classes * sizeof(float));
    cudaMalloc((void **)&d_preds, batch_size * n_classes * sizeof(float));
    cudaMalloc((void **)&d_actuals, batch_size * n_classes * sizeof(float));
    cudaMalloc((void **)&d_actuals_idx, batch_size * sizeof(float));
    cudaMalloc((void **)&d_losses, batch_size * sizeof(float));

    // Initialize weights GPU
    dim3 dimGrid = dim3(ceil(batch_size/(float)BLOCK_SIZE), ceil(n_classes/(float)BLOCK_SIZE), 1);
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    // Actual labels will be 0 or 1
    // init_vector(int *output, int rows, int n){
    init_vector<<<ceil(batch_size/(float)BLOCK_SIZE), BLOCK_SIZE>>>(d_actuals_idx, batch_size, n_classes);
    init_matrix_one_hot<<<dimGrid, BLOCK_SIZE>>>(d_actuals, d_actuals_idx, batch_size, n_classes);
    
    // Neural Network outputs will be logits
    init_matrix_normal<<<dimGrid, dimBlock>>>(d_logits, batch_size, n_classes);

    // Calculate the output probabilities
    softmax<<<dimGrid, dimBlock>>>(d_logits, d_preds, batch_size, n_classes);

    // Calculate cross entropy losses
    cross_entropy<<<dimGrid, dimBlock>>>(d_preds, d_actuals, d_losses, batch_size, n_classes);
    
    // Copy to CPU
    float *h_preds = new float[batch_size * n_classes];
    float *h_actuals_idx = new float[batch_size];
    float *h_actuals = new float[batch_size * n_classes];
    float *h_losses = new float[batch_size];
    cudaMemcpy(h_preds, d_preds, batch_size * n_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_actuals, d_actuals, batch_size * n_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_actuals_idx, d_actuals_idx, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_losses, d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Inspect outputs
    print_matrix(h_logits, batch_size, n_classes, "Logits");
    print_matrix(h_preds, batch_size, n_classes, "Preds");
    print_matrix(h_actuals_idx, batch_size, 1, "Actual Class Labels");
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
