// This program computes matrix multiplication using shared memory tiling
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include <nvToolsExt.h>

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size 
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

__global__ void tiledMatrixMul(const int *a, const int *b, int *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  nvtxRangePush("allocate host memory for three matrices");
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);
  nvtxRangePop();

  // Initialize matrices
  nvtxRangePush("initialize two source matrices with random numbers");
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
  nvtxRangePop();

  // Allocate device memory
  nvtxRangePush("allocate device memory for three matrices");
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy matrices from host to device memory");
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  std::cout << "Launch Kernel: " << THREADS << " threads per block, " << BLOCKS << " blocks in the grid" << std::endl;
  nvtxRangePush("start kernel");
  tiledMatrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  nvtxRangePop();

  // Copy back to the host
  nvtxRangePush("copy matrix from device to host memory");
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  // Check result
  nvtxRangePush("verify result");
  verify_result(h_a, h_b, h_c);
  nvtxRangePop();

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  nvtxRangePush("free device memory");
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  nvtxRangePop();

  return 0;
}
