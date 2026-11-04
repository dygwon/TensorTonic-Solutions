#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        if (x >= 0.0) output[idx] = x;
        else output[idx] = alpha * x;
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}