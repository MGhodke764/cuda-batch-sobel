#ifndef CUDA_BATCH_SOBEL_INCLUDE_IMAGE_UTILS_H_
#define CUDA_BATCH_SOBEL_INCLUDE_IMAGE_UTILS_H_

#include <string>
#include <vector>

struct PgmImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

bool ReadPgm(const std::string& path, PgmImage* image, std::string* error);
bool WritePgm(const std::string& path,
              int width,
              int height,
              const unsigned char* pixels,
              std::string* error);

#endif  // CUDA_BATCH_SOBEL_INCLUDE_IMAGE_UTILS_H_
