#ifndef CUDA_BATCH_SOBEL_INCLUDE_SOBEL_CUH_
#define CUDA_BATCH_SOBEL_INCLUDE_SOBEL_CUH_

#include <cstddef>

void LaunchSobelBatch(const unsigned char* d_input,
                      unsigned char* d_output,
                      int width,
                      int height,
                      int batch_size,
                      int block_size,
                      void* stream);

#endif  // CUDA_BATCH_SOBEL_INCLUDE_SOBEL_CUH_
