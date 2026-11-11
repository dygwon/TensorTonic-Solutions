#include <cuda_runtime.h>
#include <cmath>

__global__ void max_kernel(const float* input, int N, float* shared) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = 0.0;
    if (idx < N) {
        x = input[idx];
    } else {
        x = -INFINITY;
    }

    __shared__ float partials[256];
    partials[threadIdx.x] = x;
    __syncthreads();
    
    int s = blockDim.x / 2;
    while (s > 0) {
        if (threadIdx.x < s) {
            partials[threadIdx.x] = fmax(partials[threadIdx.x], partials[threadIdx.x + s]);
        }
        __syncthreads();
        s >>= 1;
    }

    if (threadIdx.x == 0) {
        shared[blockIdx.x] = partials[threadIdx.x];
    }
}

__global__ void global_max_kernel(const float* partials, int count, float* global_max) {
    float gmax = -INFINITY;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        gmax = fmax(partials[i], gmax);
    }

    __shared__ float shared[256];
    shared[threadIdx.x] = gmax;
    __syncthreads();
    
    int s = blockDim.x / 2;
    while (s > 0) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
        s >>= 1;
    }

    if (threadIdx.x == 0) {
        global_max[0] = shared[threadIdx.x];
    }
}

__global__ void exp_kernel(const float* input, int N, float* shared, float* global_max) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = 0.0;
    if (idx < N) {
        x = expf(input[idx] - global_max[0]);
    }

    __shared__ float partials[256];
    partials[threadIdx.x] = x;
    __syncthreads();

    int s = blockDim.x / 2;
    while (s > 0) {
        if (threadIdx.x < s) {
            partials[threadIdx.x] += partials[threadIdx.x + s];
        }
        __syncthreads();
        s >>= 1;
    }

    if (threadIdx.x == 0) {
        shared[blockIdx.x] = partials[threadIdx.x];
    }
}

__global__ void global_sum_kernel(const float* partials, int count, float* global_sum) {
    float gsum = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        gsum += partials[i];
    }

    __shared__ float shared[256];
    shared[threadIdx.x] = gsum;
    __syncthreads();

    int s = blockDim.x / 2;
    while (s > 0) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
        s >>= 1;
    }

    if (threadIdx.x == 0) {
        global_sum[0] = shared[threadIdx.x];
    }
}

__global__ void norm_kernel(const float* input, float* output, int N, float* global_max, float* global_sum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float m = global_max[0];
    float gs = global_sum[0];

    if (idx < N) {
        float x = input[idx];
        output[idx] = expf(x - m) / gs;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* shared = nullptr;
    cudaMalloc(&shared, blocks * sizeof(float));
    
    max_kernel<<<blocks, threads>>>(input, N, shared);

    float* global_max = nullptr;
    cudaMalloc(&global_max, sizeof(float));
    
    global_max_kernel<<<1, threads>>>(shared, blocks, global_max);

    float* exp_shared = nullptr;
    cudaMalloc(&exp_shared, blocks * sizeof(float));

    exp_kernel<<<blocks, threads>>>(input, N, exp_shared, global_max);

    float* global_sum = nullptr;
    cudaMalloc(&global_sum, sizeof(float));

    global_sum_kernel<<<1, threads>>>(exp_shared, blocks, global_sum);

    norm_kernel<<<blocks, threads>>>(input, output, N, global_max, global_sum);

    cudaDeviceSynchronize();

    cudaFree(shared);
    cudaFree(global_max);
    cudaFree(exp_shared);
    cudaFree(global_sum);
}