#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = (x / 2.0f) * (1.0f + erff(x / sqrtf(2.0f)));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
