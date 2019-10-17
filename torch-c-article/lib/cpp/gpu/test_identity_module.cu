#include <cstdio>
#include <cassert>
#include "cuda_helper.h"
#include "test_identity_module.h"

__global__ void kernel_identity(const float* d_input, float* d_output) {
  //const int batch_size = gridDim.x;
  const int channels = gridDim.y;
  const int height = blockDim.x;
  const int width = blockDim.y;
  const int depth = blockDim.z;

  const int b = blockIdx.x;
  const int c = blockIdx.y;
  const int h = threadIdx.x;
  const int w = threadIdx.y;
  const int d = threadIdx.z;

  d_output[(((b*channels + c)*height + h)*width + w)*depth + d] = d_input[(((b*channels + c)*height + h)*width + w)*depth + d];
}

void test_identity_module_updateOutput(const int rank, const long* dims, const float* d_input, float* d_output) {
  assert(rank == 5);

  const int batch_size = dims[0];
  const int channels = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int depth = dims[4];

  dim3 grid(batch_size, channels, 1);
  dim3 block(height, width, depth);

  kernel_identity<<<grid, block>>>(d_input, d_output);
}

void test_identity_module_updateGradInput(const int rank, const long* dims, const float* d_input, const float* d_grad_output, float* d_grad_input) {
  assert(rank == 5);

  const int batch_size = dims[0];
  const int channels = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int depth = dims[4];

  dim3 grid(batch_size, channels, 1);
  dim3 block(height, width, depth);

  kernel_identity<<<grid, block>>>(d_grad_output, d_grad_input);
}