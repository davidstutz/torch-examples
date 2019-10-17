#include <cstdio>
#include <cassert>
#include "test_module.h"

void test_module_updateOutput(const int rank, const long* dims, const float* input, float* output) {
  assert(rank == 5);

  for (int r = 0; r < rank; r++) {
    printf("%ld ", dims[r]);
  }
  printf("\n");

  const int batch_size = dims[0];
  const int channels = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int depth = dims[4];

  for (int b = 0; b < batch_size; b++) {
    printf("batch %d\n", b);
    for (int c = 0; c < channels; c++) {
      printf("channel %d\n", c);
      for (int h = 0; h < height; h++) {
        printf("height %d\n", h);
        for (int w = 0; w < width; w++) {
          for (int d = 0; d < depth; d++) {
            printf("%f ", input[(((b*channels + c)*height + h)*width + w)*depth + d]);
          }
          printf("\n");
        }
      }
    }
  }
}

void test_module_updateGradInput(const int rank, const long* dims, const float* input, const float* grad_output, float* grad_input) {
  assert(rank == 5);

  for (int r = 0; r < rank; r++) {
    printf("%ld ", dims[r]);
  }
  printf("\n");

  const int batch_size = dims[0];
  const int channels = dims[1];
  const int height = dims[2];
  const int width = dims[3];
  const int depth = dims[4];

  for (int b = 0; b < batch_size; b++) {
    printf("batch %d\n", b);
    for (int c = 0; c < channels; c++) {
      printf("channel %d\n", c);
      for (int h = 0; h < height; h++) {
        printf("height %d\n", h);
        for (int w = 0; w < width; w++) {
          for (int d = 0; d < depth; d++) {
            printf("%f ", input[(((b*channels + c)*height + h)*width + w)*depth + d]);
          }
          printf("\n");
        }
      }
    }
  }
}