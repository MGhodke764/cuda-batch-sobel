# CUDA-Accelerated Batch Sobel Edge Detection

## Overview

This project demonstrates CUDA image processing at scale by applying Sobel edge detection to hundreds of grayscale PGM images in a single GPU batch.

The project uses a custom CUDA kernel. Each CUDA thread is responsible for one output pixel, while shared memory is used as a tile cache around each thread block. The batch dimension is mapped to the CUDA grid's `z` dimension, allowing many independent images to be processed by the same kernel launch.

The dataset is generated locally with a deterministic Python script, so no external dataset download is required.

## Features

- Custom CUDA Sobel edge-detection kernel
- Batch processing of hundreds of images
- Shared-memory image tiles
- Command-line arguments
- CUDA event timing for kernel performance
- CSV performance output
- Reproducible synthetic PGM dataset
- Makefile for compilation
- `run.sh` for end-to-end execution
- No third-party image library required

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (`nvcc`)
- C++17 compiler support
- Python 3
- Linux environment recommended

The Coursera CUDA laboratory environment is the intended execution environment.

## Project structure

```text
cuda-batch-sobel/
├── include/
│   ├── image_utils.h
│   └── sobel.cuh
├── src/
│   ├── image_utils.cpp
│   ├── main.cu
│   └── sobel.cu
├── scripts/
│   └── generate_dataset.py
├── data/
│   └── images/
├── results/
├── Makefile
├── run.sh
└── README.md
```

## Generate the dataset

Generate 500 images at 256 x 256 pixels:

```bash
python3 scripts/generate_dataset.py \
  --output data/images \
  --count 500 \
  --width 256 \
  --height 256
```

The generator uses only Python's standard library.

## Build

```bash
make
```

Clean the build:

```bash
make clean
```

## Run

Run the complete 500-image experiment:

```bash
./run.sh
```

Or specify the number of images and CUDA block size:

```bash
./run.sh 500 16
```

The lower-level executable also accepts arguments directly:

```bash
./cuda_sobel \
  --input data/images \
  --output results/edges \
  --limit 500 \
  --block 16
```

The current implementation uses a 16 x 16 CUDA block because its shared-memory tile is sized for a 16 x 16 block plus a one-pixel halo.

## CUDA implementation

The Sobel operator computes horizontal and vertical gradients:

```text
Gx = [-1  0  1]       Gy = [-1 -2 -1]
     [-2  0  2]            [ 0  0  0]
     [-1  0  1]            [ 1  2  1]
```

The implementation uses:

1. Global memory for the input and output image batches.
2. Shared memory for the pixel tile used by a CUDA block.
3. One CUDA thread per output pixel.
4. `blockIdx.z` to select the image being processed.
5. CUDA events to measure kernel execution time.

This makes the image batch naturally parallel because the pixels of different images do not depend on each other.

## Execution artifacts

After execution, the `results/` directory contains:

- `execution_log.txt` — GPU information, workload size, and timing.
- `performance.csv` — measured kernel and total processing time.
- `edges/` — Sobel output images.

For peer review, these files can be compressed into a ZIP file and uploaded to Coursera as proof of execution.

## Results interpretation

The most important measurement is `GPU kernel time`, which measures the CUDA Sobel computation using CUDA events. `Total processing time` includes the GPU computation and related processing performed by the application.

The actual timing values depend on the GPU used by the Coursera laboratory environment, so results are recorded from the execution rather than hard-coded.

## Lessons learned

The project demonstrates that image processing is well suited to GPU parallelism because individual pixels and independent images can be processed concurrently. Shared memory can reduce repeated global-memory reads for neighboring pixels, which is useful for convolution-style operations such as Sobel filtering.

A key lesson is that application-level performance includes memory transfers and file I/O, not only kernel execution. Therefore, GPU kernel timing and total application timing should be considered separately.

## License

This project is provided for educational use as part of the CUDA at Scale for the Enterprise course.
