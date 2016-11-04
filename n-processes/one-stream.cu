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
__global__ void slow_kernel(int *dst, int *src, int n) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  #define DELAY 2048
  for (volatile int i = 0; i < DELAY; i++)
    ;
  if (id < n) {
    dst[id] = src[id];
  }
}

__global__ void fast_kernel(int *dst, int *src, int n) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    dst[id] = src[id];
  }
}

// initialize memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n; i++) {
    buf[i] = i;
  }
}

// Forward declarations
cudaError_t setup_memory(int* src[], int* dst[], size_t size, size_t n_regions);
cudaError_t iterate_stream(bool delay, size_t iterations, int* src[],
    int* dst[], size_t n_regions);

int main(int argc, char **argv) {
  cudaDeviceProp device_prop;
  int dev_id;

  printf("Starting [%s]...\n", argv[0]);

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
  const bool delay = checkCmdLineFlag(argc, (const char **) argv, "delay");
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
        "Please provide --size=<int> --priority=<int> --iterations=<int> and --delay(optional) flags.\nexting...\n");
    exit (EXIT_FAILURE);
  }
  
  if (priority_hi > priority || priority_low < priority) {
    fprintf(stderr, "Priority must be within %d and %d.\nexting...\n",
        priority_hi, priority_low);
    exit (EXIT_FAILURE);
  }

  // Create memory regions
#define N_MEMORY_REGIONS 8
  size_t n_regions = max(min(N_MEMORY_REGIONS, iterations / 4), 1);
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
  if (delay) {
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, slow_kernel);
  } else {
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, fast_kernel);    
  }
  gridSize = (size + blockSize - 1) / blockSize;
 
  // launch the kernel iteration times.
  // each consecutive launch uses a different memory region
  for (int i = 0; i < ROUND_UP(iterations, n_regions); i++) {
    for (int j = 0; j < n_regions; j++) {
      if (delay) {
        slow_kernel<<<gridSize, blockSize, 0, stream>>>(dst[j], src[j], size);
      } else {
        fast_kernel<<<gridSize, blockSize, 0, stream>>>(dst[j], src[j], size);
      }
    }
  }

  checkCudaErrors(cudaStreamSynchronize(stream));

  // check results of the last computation
  for (int i = 0; i < n_regions; i++) {
    ERR_NE(memcmp(dst[i], src[i], size), 0);  
  }
  
  // Clean up
  for (int i = 0; i < n_regions; i++) {
    checkCudaErrors (cudaFree(src[i]));
    checkCudaErrors(cudaFree(dst[i]));
  }
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

/* */
