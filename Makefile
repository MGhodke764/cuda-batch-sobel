CUDA_ARCH ?= native
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Iinclude -Xcompiler -Wall,-Wextra
NVCCFLAGS := -std=c++17 -O2 -Iinclude

TARGET := cuda_sobel
BUILD_DIR := build

all: $(TARGET)

$(TARGET): $(BUILD_DIR)/main.o $(BUILD_DIR)/sobel.o $(BUILD_DIR)/image_utils.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main.o: src/main.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -Wall,-Wextra -c $< -o $@

$(BUILD_DIR)/sobel.o: src/sobel.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -Wall,-Wextra -c $< -o $@

$(BUILD_DIR)/image_utils.o: src/image_utils.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -Wall,-Wextra -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
