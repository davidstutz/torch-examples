#include <cstdio>
#include <cassert>
#include "test_identity_module.h"

void test_identity_module_updateOutput(const int rank, const long* dims, const float* input, float* output) {
  assert(rank == 5);

  const int batch_size = dims[0];
  const int channels = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int depth = dims[4];

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          for (int d = 0; d < depth; d++) {
            output[(((b*channels + c)*height + h)*width + w)*depth + d] = input[(((b*channels + c)*height + h)*width + w)*depth + d];
          }
        }
      }
    }
  }
}

void test_identity_module_updateGradInput(const int rank, const long* dims, const float* input, const float* grad_output, float* grad_input) {
  assert(rank == 5);

  const int batch_size = dims[0];
  const int channels = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int depth = dims[4];

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          for (int d = 0; d < depth; d++) {
            grad_input[(((b*channels + c)*height + h)*width + w)*depth + d] = grad_output[(((b*channels + c)*height + h)*width + w)*depth + d];
          }
        }
      }
    }
  }
}