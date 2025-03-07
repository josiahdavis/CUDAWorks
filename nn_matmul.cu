#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

// He i.e., normal initialization
__global__ void init_rand(int w, int h, float* mat)
{
    int column = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < h && column < w){
        curandState state;
        curand_init(42, row * w + column, 0, &state);
        mat[row*w + column] = curand_normal(&state)*sqrtf(2.f/h);
    }
}

// Computes output = W * X + b
__global__ void matmul(int batch_size, int in_w, int out_w, float* input, 
                        float* weights, float* biases, float* output)
{
    int column = blockIdx.x*blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && column < out_w)
    {
        output[row*out_w + column] = biases[column];
        for(int i = 0; i < in_w; i++)
        {
            output[row * out_w + column] += weights[i * out_w + column] * input[row * in_w + i];
        }
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* name){
    printf("Matrix %s showing top-left 5x5\n", name);
    for (int i = 0; i < 5 && i < rows; i++){
        for (int j = 0; j < 5 && j < cols; j++){
            printf("%5.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(){

    // Set up parameters
    int batch_size = 2;
    int in_features = 3;
    int out_features = 4;
    int BLOCK_SIZE = 16;
    
    // Allocate memory for GPU
    float *d_X;
    float *d_weights;
    float *d_biases;
    float *d_out;
    cudaMalloc((void **)&d_X, batch_size * in_features * sizeof(float));
    cudaMalloc((void **)&d_weights, in_features * out_features * sizeof(float));
    cudaMalloc((void **)&d_biases, out_features * sizeof(float));
    cudaMalloc((void **)&d_out, batch_size * out_features * sizeof(float));

    // Initialize weights GPU
    dim3 dimGrid = dim3(ceil(out_features/(float)BLOCK_SIZE), ceil(in_features/(float)BLOCK_SIZE), 1);
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    init_rand<<<dimGrid, dimBlock>>>(out_features, in_features, d_weights);
    
    // Initialize biases GPU
    dimGrid = dim3(ceil(out_features/(float)BLOCK_SIZE), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    init_rand<<<dimGrid, dimBlock>>>(out_features, 1, d_biases);

    // Initialize data GPU
    dimGrid = dim3(ceil(in_features/(float)BLOCK_SIZE), ceil(batch_size/(float)BLOCK_SIZE), 1);
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    init_rand<<<dimGrid, dimBlock>>>(in_features, batch_size, d_X);

    // Perform Matrix Multiplication
    dimGrid = dim3(ceil(out_features/(float)BLOCK_SIZE), ceil(batch_size/(float)BLOCK_SIZE), 1);
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    // batch_size, n, out_w, input, weights, biases, output
    matmul<<<dimGrid, dimBlock>>>(batch_size, in_features, out_features, d_X, d_weights, d_biases, d_out);

    // Copy data to CPU
    float *h_X = new float[batch_size * in_features];
    float *h_weights = new float[in_features * out_features];
    float *h_out = new float[batch_size * out_features];
    float *h_biases = new float[out_features];
    cudaMemcpy(h_X, d_X, batch_size * in_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weights, d_weights, in_features * out_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_biases, d_biases, out_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Print data for inspection
    print_matrix(h_X, batch_size, in_features, "X");
    print_matrix(h_weights, in_features, out_features, "Weights");
    print_matrix(h_biases, 1, out_features, "Biases");
    print_matrix(h_out, batch_size, out_features, "Output");
    return 0;
}