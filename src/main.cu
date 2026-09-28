#include "image_utils.h"
#include "sobel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_dir = "data/images";
    std::string output_dir = "results/edges";
    int limit = 500;
    int block_size = 16;
};

void PrintUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --input <dir>       Input PGM image directory\n"
        << "  --output <dir>      Output directory\n"
        << "  --limit <N>         Maximum images to process (default: 500)\n"
        << "  --block <N>         CUDA block dimension (currently 16)\n"
        << "  --help              Show this message\n";
}

bool ParseInt(const std::string& value, int* result) {
    try {
        std::size_t position = 0;
        const int parsed = std::stoi(value, &position);
        if (position != value.size()) {
            return false;
        }
        *result = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool ParseOptions(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help") {
            PrintUsage(argv[0]);
            return false;
        }

        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << "\n";
            return false;
        }

        if (arg == "--input") {
            options->input_dir = argv[++i];
        } else if (arg == "--output") {
            options->output_dir = argv[++i];
        } else if (arg == "--limit") {
            if (!ParseInt(argv[++i], &options->limit)) {
                std::cerr << "Invalid --limit value\n";
                return false;
            }
        } else if (arg == "--block") {
            if (!ParseInt(argv[++i], &options->block_size)) {
                std::cerr << "Invalid --block value\n";
                return false;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            PrintUsage(argv[0]);
            return false;
        }
    }

    if (options->limit <= 0) {
        std::cerr << "--limit must be positive\n";
        return false;
    }

    if (options->block_size != 16) {
        std::cerr << "--block must currently be 16 because the shared-memory "
                     "tile is 16 x 16 plus a one-pixel halo.\n";
        return false;
    }

    return true;
}

bool IsPgm(const fs::path& path) {
    return path.extension() == ".pgm" || path.extension() == ".PGM";
}

std::vector<fs::path> CollectImages(const std::string& directory, int limit) {
    std::vector<fs::path> paths;

    if (!fs::exists(directory)) {
        return paths;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && IsPgm(entry.path())) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());

    if (static_cast<int>(paths.size()) > limit) {
        paths.resize(limit);
    }

    return paths;
}

void CheckCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error during " << operation << ": "
                  << cudaGetErrorString(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        return 1;
    }

    int device_count = 0;
    CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU was detected.\n";
        return 1;
    }

    cudaDeviceProp properties{};
    CheckCuda(cudaGetDeviceProperties(&properties, 0),
              "cudaGetDeviceProperties");

    const std::vector<fs::path> paths =
        CollectImages(options.input_dir, options.limit);

    if (paths.empty()) {
        std::cerr << "No PGM images found in " << options.input_dir << "\n";
        return 1;
    }

    std::vector<PgmImage> images;
    images.reserve(paths.size());

    for (const auto& path : paths) {
        PgmImage image;
        std::string error;
        if (!ReadPgm(path.string(), &image, &error)) {
            std::cerr << "Skipping image: " << error << "\n";
            continue;
        }

        if (!images.empty() &&
            (image.width != images.front().width ||
             image.height != images.front().height)) {
            std::cerr << "Skipping image with different dimensions: "
                      << path << "\n";
            continue;
        }

        images.push_back(std::move(image));
    }

    if (images.empty()) {
        std::cerr << "No compatible PGM images were loaded.\n";
        return 1;
    }

    const int width = images.front().width;
    const int height = images.front().height;
    const std::size_t pixels_per_image =
        static_cast<std::size_t>(width) * height;
    const std::size_t total_pixels =
        pixels_per_image * images.size();
    const std::size_t buffer_bytes =
        total_pixels * sizeof(unsigned char);

    std::vector<unsigned char> input(total_pixels);
    std::vector<unsigned char> output(total_pixels);

    for (std::size_t i = 0; i < images.size(); ++i) {
        std::copy(images[i].pixels.begin(),
                  images[i].pixels.end(),
                  input.begin() + static_cast<std::ptrdiff_t>(
                      i * pixels_per_image));
    }

    fs::create_directories(options.output_dir);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    cudaStream_t stream = nullptr;

    CheckCuda(cudaMalloc(&d_input, buffer_bytes), "cudaMalloc input");
    CheckCuda(cudaMalloc(&d_output, buffer_bytes), "cudaMalloc output");
    CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    std::cout << "CUDA Batch Sobel Edge Detection\n"
              << "================================\n"
              << "GPU: " << properties.name << "\n"
              << "Compute capability: " << properties.major << "."
              << properties.minor << "\n"
              << "Input directory: " << options.input_dir << "\n"
              << "Output directory: " << options.output_dir << "\n"
              << "Images processed: " << images.size() << "\n"
              << "Image size: " << width << " x " << height << "\n"
              << "Total pixels: " << total_pixels << "\n"
              << "CUDA block: " << options.block_size << " x "
              << options.block_size << "\n"
              << "Device memory per buffer: "
              << std::fixed << std::setprecision(2)
              << buffer_bytes / (1024.0 * 1024.0) << " MiB\n\n";

    CheckCuda(cudaMemcpyAsync(d_input, input.data(), buffer_bytes,
                              cudaMemcpyHostToDevice, stream),
              "host-to-device copy");

    CheckCuda(cudaStreamSynchronize(stream), "input copy synchronization");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate start");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord start");

    LaunchSobelBatch(d_input, d_output, width, height,
                     static_cast<int>(images.size()),
                     options.block_size, stream);

    CheckCuda(cudaGetLastError(), "Sobel kernel launch");

    CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    CheckCuda(cudaEventSynchronize(stop), "kernel synchronization");

    float kernel_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&kernel_ms, start, stop),
              "cudaEventElapsedTime");

    const auto total_start = std::chrono::steady_clock::now();

    CheckCuda(cudaMemcpyAsync(output.data(), d_output, buffer_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "device-to-host copy");
    CheckCuda(cudaStreamSynchronize(stream), "output copy synchronization");

    const auto total_end = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < images.size(); ++i) {
        const fs::path output_path =
            fs::path(options.output_dir) /
            (paths[i].stem().string() + "_edges.pgm");

        std::string error;
        if (!WritePgm(
                output_path.string(),
                width,
                height,
                output.data() + static_cast<std::ptrdiff_t>(
                    i * pixels_per_image),
                &error)) {
            std::cerr << error << "\n";
        }
    }

    const double total_ms =
        std::chrono::duration<double, std::milli>(
            total_end - total_start).count() + kernel_ms;

    const double average_ms =
        kernel_ms / static_cast<double>(images.size());

    const fs::path csv_path =
        fs::path(options.output_dir) / "performance.csv";
    std::ofstream csv(csv_path);
    csv << "image_count,width,height,block_x,block_y,kernel_ms,"
           "total_ms,average_kernel_ms\n";
    csv << images.size() << ','
        << width << ','
        << height << ','
        << options.block_size << ','
        << options.block_size << ','
        << std::fixed << std::setprecision(4)
        << kernel_ms << ','
        << total_ms << ','
        << average_ms << '\n';

    std::cout << "\nExecution completed successfully.\n"
              << "GPU kernel time: " << std::fixed
              << std::setprecision(4) << kernel_ms << " ms\n"
              << "Total processing time: " << total_ms << " ms\n"
              << "Average kernel time/image: " << average_ms << " ms\n"
              << "Output images: " << options.output_dir << "\n"
              << "Performance CSV: " << csv_path << "\n";

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    CheckCuda(cudaFree(d_input), "cudaFree input");
    CheckCuda(cudaFree(d_output), "cudaFree output");

    return 0;
}
