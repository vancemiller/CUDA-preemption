/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// std::system includes
#include <cstdio>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_string.h>

// Semaphore include
#include <sys/types.h>
#include <unistd.h>

// throw error on equality
#define ERR_EQ(X,Y) do { if ((X) == (Y)) { \
  fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
  exit(-1);}} while(0)

// throw error on difference
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
  fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
  exit(-1);}} while(0)

#define ROUND_UP(N, BASE) \
  (N + BASE - 1) / BASE

// copy from source -> destination arrays
__device__ void slow_kernel(int *dst, int *src, int n, int delay) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (volatile int i = 0; i < delay; i++);
  if (id < n) {
    dst[id] = src[id];
  }
}

// Named kernels for easier profiling
__global__ void low_priority(int *dst, int *src, int n, int delay) {
  slow_kernel(dst, src, n, delay);
}

__global__ void high_priority(int *dst, int *src, int n, int delay) {
  slow_kernel(dst, src, n, delay);
}

// initialize memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n; i++) {
    buf[i] = i;
  }
}

// Forward declarations
cudaError_t setup_memory(int* src[], int* dst[], size_t size, size_t n_regions);
void run_experiment(const int priority, const int size, const int iterations,
    const int delay);

int main(int argc, char **argv) {
  cudaDeviceProp device_prop;
  int dev_id;

  fprintf(stderr, "Starting [%s]...\n", argv[0]);

  // set device
  dev_id = findCudaDevice(argc, (const char **) argv);
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
  if ((device_prop.major << 4) + device_prop.minor < 0x35) {
    fprintf(stderr,
        "%s requires Compute Capability of SM 3.5 or higher to run.\nexiting...\n",
        argv[0]);
    exit (EXIT_WAIVED);
  }

  // command line args
  const int delay = getCmdLineArgumentInt(argc, (const char **) argv, "delay");
  const size_t size = getCmdLineArgumentInt(argc, (const char **) argv, "size");
  const int priority = getCmdLineArgumentInt(argc, (const char **) argv, "priority");
  const int iterations = getCmdLineArgumentInt(argc, (const char **) argv, "iterations");

  // get the range of priorities available
  // [ greatest_priority, least_priority ]
  int priority_low;
  int priority_hi;
  checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low,
      &priority_hi));

  if (size == 0 || iterations == 0) {
    fprintf(stderr,
        "Please provide --size=<int> --priority=<int> --iterations=<int> "
        "and --delay=<int> (optional) flags.\nexting...\n");
    exit (EXIT_FAILURE);
  } else {
    fprintf(stderr, "Called with arguments size %zu, priority %d, iterations %d, and delay %d\n",
        size, priority, iterations, delay);
  }
  
  if (priority_hi > priority || priority_low < priority) {
    fprintf(stderr, "Priority must be within %d and %d.\nexting...\n",
        priority_hi, priority_low);
    exit (EXIT_FAILURE);
  }

  // Set kernel to run
  void (*kernel)(int*, int*, int, int) = priority ? &high_priority : &low_priority;
  
  // Create memory regions
#define N_MEMORY_REGIONS 8
  size_t n_regions = N_MEMORY_REGIONS;
  int *src[n_regions];
  int *dst[n_regions];
  
  setup_memory(src, dst, size, n_regions);
  cudaDeviceSynchronize();

  // Create stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithPriority(&stream,
      cudaStreamNonBlocking, priority));
 
  // Compute number of threads and blocks
  int blockSize;
  int minGridSize;
  int gridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
  gridSize = (size + blockSize - 1) / blockSize;
 
  // launch the kernel iteration times.
  // each consecutive launch uses a different memory region
  cudaEvent_t start, end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));
  checkCudaErrors(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; i++) {
    for (int j = 0; j < n_regions; j++, i++) {
      if (i >= iterations) {
        break;
      }
      kernel<<<gridSize, blockSize, 0, stream>>>(dst[j], src[j], size,
          delay);
      checkCudaErrors(cudaStreamSynchronize(stream));
    }
  }
  checkCudaErrors(cudaEventRecord(end, stream));
  checkCudaErrors(cudaEventSynchronize(end));

  // check results of the last computation
  for (int i = 0; i < n_regions && i < iterations; i++) {
    ERR_NE(memcmp(dst[i], src[i], size), 0);  
  }
  
  // Clean up
  for (int i = 0; i < n_regions; i++) {
    checkCudaErrors (cudaFree(src[i]));
    checkCudaErrors(cudaFree(dst[i]));
  }
  // Print out average time
  float ms;
  checkCudaErrors(cudaEventElapsedTime(&ms, start, end));

  // size iterations ms average
  printf("%zu, %d, %f, %f\n", size, iterations, ms, ms / (float) iterations);
  exit (EXIT_SUCCESS);
}

cudaError_t setup_memory(int* src[], int* dst[], size_t size,
    size_t n_regions) {
  for (int i = 0; i < n_regions; i++) {
    checkCudaErrors(cudaMallocManaged(&src[i], size * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&dst[i], size * sizeof(int)));
    mem_init(src[i], size);
    memset(dst[i], 0, size);
  }
  return cudaSuccess;
}

