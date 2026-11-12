#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    // Write code here
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (r < M && c < N) {
        int inIdx = r * N + c;
        int outIdx = c * M + r;
        B[outIdx] = A[inIdx];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
