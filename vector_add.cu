#include <stdio.h>
#include <math.h>
#include <time.h>         
                                                     
void init_vector(float *v, int size, float bound){
    for (int i = 0; i < size; i++){
        v[i] = -bound + bound * 2.0f * (float)rand() / (float)RAND_MAX;
    }                            
}                     
                                                     
void head(float *v){     
    printf("[");   
    for (int i = 0; i < 10; i++){
        printf("%7.2f ", v[i]);
    }       
    printf("]\n");
}                                   
                                                     
__global__ void add(int n, float* a, float* b, float* c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){                                
        c[i] = b[i] + a[i];                    
    }                                          
}

int main(){
    int N = 2048;
    // Allocate c pointers
    float* a = new float[N];
    float* b = new float[N];
    float* c = new float[N];
    // Initialize vectors on host
    srand(time(NULL));
    init_vector(a, N, 100);
    init_vector(b, N, 1);
    printf("a = ");
    head(a);
    printf("b = ");
    head(b);
    // Allocate memory on the device
    float* a_d;
    float* b_d;
    float* c_d;
    cudaMalloc((void**) &a_d, N*sizeof(float));
    cudaMalloc((void**) &b_d, N*sizeof(float));
    cudaMalloc((void**) &c_d, N*sizeof(float));
    // Copy data from the CPU to the GPU
    cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, N*sizeof(float), cudaMemcpyHostToDevice);
    // Run the kernel
    int N_BLOCKS=4;
    int N_THREADS=1024;
    add<<<N_BLOCKS, N_THREADS>>>(N, a_d, b_d, c_d);
    // Copy data back to CPU
    cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    // Inspect output
    printf("c = ");
    head(c);
    // Clean up memory
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}