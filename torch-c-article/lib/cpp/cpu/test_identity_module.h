#ifndef CPU_TEST_SUM_MODULE
#define CPU_TEST_SUM_MODULE

extern "C" {
  void test_identity_module_updateOutput(const int rank, const long* dims, const float* input, float* output);
  void test_identity_module_updateGradInput(const int rank, const long* dims, const float* input, const float* grad_output, float* grad_input);
}

#endif