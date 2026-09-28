#include "sobel.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void SobelBatchKernel(const unsigned char* input,
                                 unsigned char* output,
                                 int width,
                                 int height) {
    constexpr int kTile = 16;
    constexpr int kRadius = 1;
    constexpr int kSharedWidth = kTile + 2 * kRadius;

    __shared__ unsigned char tile[kSharedWidth][kSharedWidth];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * kTile + tx;
    const int y = blockIdx.y * kTile + ty;
    const int image = blockIdx.z;

    const std::size_t image_offset =
        static_cast<std::size_t>(image) * width * height;

    const int sx = tx + kRadius;
    const int sy = ty + kRadius;

    if (x < width && y < height) {
        tile[sy][sx] = input[image_offset +
                             static_cast<std::size_t>(y) * width + x];
    } else {
        tile[sy][sx] = 0;
    }

    if (tx < kRadius) {
        const int left_x = x - kRadius;
        tile[sy][tx] =
            (left_x >= 0 && y < height)
                ? input[image_offset +
                        static_cast<std::size_t>(y) * width + left_x]
                : 0;

        const int right_x = x + kTile;
        tile[sy][sx + kTile] =
            (right_x < width && y < height)
                ? input[image_offset +
                        static_cast<std::size_t>(y) * width + right_x]
                : 0;
    }

    if (ty < kRadius) {
        const int top_y = y - kRadius;
        tile[ty][sx] =
            (top_y >= 0 && x < width)
                ? input[image_offset +
                        static_cast<std::size_t>(top_y) * width + x]
                : 0;

        const int bottom_y = y + kTile;
        tile[sy + kTile][sx] =
            (bottom_y < height && x < width)
                ? input[image_offset +
                        static_cast<std::size_t>(bottom_y) * width + x]
                : 0;
    }

    if (tx < kRadius && ty < kRadius) {
        const int top_y = y - kRadius;
        const int left_x = x - kRadius;
        tile[ty][tx] =
            (top_y >= 0 && left_x >= 0)
                ? input[image_offset +
                        static_cast<std::size_t>(top_y) * width + left_x]
                : 0;

        const int top_right_x = x + kTile;
        tile[ty][sx + kTile] =
            (top_y >= 0 && top_right_x < width)
                ? input[image_offset +
                        static_cast<std::size_t>(top_y) * width + top_right_x]
                : 0;

        const int bottom_left_y = y + kTile;
        tile[sy + kTile][tx] =
            (bottom_left_y < height && left_x >= 0)
                ? input[image_offset +
                        static_cast<std::size_t>(bottom_left_y) * width +
                        left_x]
                : 0;

        const int bottom_right_y = y + kTile;
        tile[sy + kTile][sx + kTile] =
            (bottom_right_y < height && top_right_x < width)
                ? input[image_offset +
                        static_cast<std::size_t>(bottom_right_y) * width +
                        top_right_x]
                : 0;
    }

    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[image_offset +
               static_cast<std::size_t>(y) * width + x] = 0;
        return;
    }

    const int gx =
        -static_cast<int>(tile[sy - 1][sx - 1]) +
        static_cast<int>(tile[sy - 1][sx + 1]) -
        2 * static_cast<int>(tile[sy][sx - 1]) +
        2 * static_cast<int>(tile[sy][sx + 1]) -
        static_cast<int>(tile[sy + 1][sx - 1]) +
        static_cast<int>(tile[sy + 1][sx + 1]);

    const int gy =
        -static_cast<int>(tile[sy - 1][sx - 1]) -
        2 * static_cast<int>(tile[sy - 1][sx]) -
        static_cast<int>(tile[sy - 1][sx + 1]) +
        static_cast<int>(tile[sy + 1][sx - 1]) +
        2 * static_cast<int>(tile[sy + 1][sx]) +
        static_cast<int>(tile[sy + 1][sx + 1]);

    const int magnitude = min(255, abs(gx) + abs(gy));

    output[image_offset +
           static_cast<std::size_t>(y) * width + x] =
        static_cast<unsigned char>(magnitude);
}

}  // namespace

void LaunchSobelBatch(const unsigned char* d_input,
                      unsigned char* d_output,
                      int width,
                      int height,
                      int batch_size,
                      int block_size,
                      void* stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const dim3 block(block_size, block_size, 1);
    const dim3 grid(
        (width + block_size - 1) / block_size,
        (height + block_size - 1) / block_size,
        batch_size);

    SobelBatchKernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height);
}
