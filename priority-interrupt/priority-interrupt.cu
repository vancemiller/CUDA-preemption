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

#define TOTAL_SIZE  256*1024*1024

// # threadblocks
#define TBLOCKS 1024
#define THREADS  512

// throw error on equality
#define ERR_EQ(X,Y) do { if ((X) == (Y)) { \
  fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
  exit(-1);}} while(0)

// throw error on difference
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
  fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
  exit(-1);}} while(0)

// experiment iterations to compute averages
#define N_ITERATIONS 2
#define N_HIGH_KERNELS 4

// copy from source -> destination arrays
__device__ void slow_kernel(int *dst, int *src, size_t n) {
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = id; i < n / sizeof(int); i += num) {
#define DELAY 2048
    for (volatile int j = 0; j < DELAY; j++)
      ;
    dst[i] = src[i];
  }
}

__device__ void fast_kernel(int *dst, int *src, size_t n) {
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = id; i < n / sizeof(int); i += num) {
    dst[i] = src[i];
  }
}

// named kernels for easier profiling
__global__ void low(int *dst, int *src, size_t n) {
  slow_kernel(dst, src, n);
}

__global__ void high(int *dst, int *src, size_t n) {
  fast_kernel(dst, src, n);
}

__global__ void low_preempt(int *dst, int *src, size_t n) {
  slow_kernel(dst, src, n);
}

__global__ void high_preempt(int *dst, int *src, size_t n) {
  fast_kernel(dst, src, n);
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

// Forward declaration
cudaError_t experiment(int priority_low, int priority_hi);

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

  // get the range of priorities available
  // [ greatest_priority, least_priority ]
  int priority_low;
  int priority_hi;
  checkCudaErrors(
      cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));
  printf("CUDA stream priority range: LOW: %d to HIGH: %d\n", priority_low,
      priority_hi);

  experiment(priority_low, priority_hi);

  exit (EXIT_SUCCESS);
}

cudaError_t solo_test(cudaStream_t* streams, int low_priority_stream_idx,
    int high_priority_stream_idx, size_t n_streams, size_t size) {
  // Initialize host data
  int *h_src[n_streams];
  for (int i = 0; i < n_streams; i++) {
    ERR_EQ(h_src[i] = (int * ) malloc(size), NULL);
    mem_init(h_src[i], size);
  }

  // Initialize device data
  int *h_dst[n_streams];
  for (int i = 0; i < n_streams; i++) {
    ERR_EQ(h_dst[i] = (int * ) malloc(size), NULL);
    memset(h_dst[i], 0, size);
  }

  // copy source data -> device
  int *d_src[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMalloc(&d_src[i], size));
    checkCudaErrors(
        cudaMemcpyAsync(d_src[i], h_src[i], size, cudaMemcpyHostToDevice, streams[i]));
  }

  // allocate memory for memcopy destination
  int *d_dst[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMalloc(&d_dst[i], size));
  }

  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaStreamSynchronize(streams[i]));
  }
  /* Kernel invocations */

  // Run each priority on its own
  for (int i = 0; i < n_streams; i++) {
    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventRecord(start, streams[i]));
    if (i == low_priority_stream_idx) {
      low<<<TBLOCKS, THREADS, 0, streams[i]>>>(d_dst[i], d_src[i], TOTAL_SIZE);
    } else {
      high<<<TBLOCKS, THREADS, 0, streams[i]>>>(d_dst[i], d_src[i], TOTAL_SIZE);
    }
    checkCudaErrors(cudaEventRecord(end, streams[i]));
    checkCudaErrors(cudaEventSynchronize(end));
  }

  // Copy result to host
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(
        cudaMemcpyAsync(h_dst[i], d_dst[i], size, cudaMemcpyDeviceToHost, streams[i]));
  }

//  // check results of kernels
//  for (int i = 0; i < n_streams; i++) {
//    ERR_NE(memcmp(h_dst[i], h_src[i], size), 0);
//  }

  // Clean up
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaFree(d_src[i]));
    checkCudaErrors(cudaFree(d_dst[i]));
  }
  for (int i = 0; i < n_streams; i++) {
    free(h_src[i]);
  }
  return cudaSuccess;
}

cudaError_t concurrent_test(cudaStream_t* streams, int low_priority_stream_idx,
    int high_priority_stream_idx, int n_kernels, size_t size) {
  // Initialize host data
  int *h_src[n_kernels];
  for (int i = 0; i < n_kernels; i++) {
    ERR_EQ(h_src[i] = (int * ) malloc(size), NULL);
    mem_init(h_src[i], size);
  }

  // Initialize device data
  int *h_dst[n_kernels];
  for (int i = 0; i < n_kernels; i++) {
    ERR_EQ(h_dst[i] = (int * ) malloc(size), NULL);
    memset(h_dst[i], 0, size);
  }

  // copy source data -> device
  int *d_src[n_kernels];
  checkCudaErrors(cudaMalloc(&d_src[0], size));
  checkCudaErrors(
      cudaMemcpyAsync(d_src[0], h_src[0], size, cudaMemcpyHostToDevice, streams[low_priority_stream_idx]));

  for (int i = 0; i < n_kernels; i++) {
    checkCudaErrors(cudaMalloc(&d_src[i], size));
    checkCudaErrors(
        cudaMemcpyAsync(d_src[i], h_src[i], size, cudaMemcpyHostToDevice, streams[high_priority_stream_idx]));
  }

  // allocate memory for memcopy destination
  int *d_dst[n_kernels];
  for (int i = 0; i < n_kernels; i++) {
    checkCudaErrors(cudaMalloc(&d_dst[i], size));
  }

  /* */

  // create some events
  cudaEvent_t ev_start[n_kernels];
  cudaEvent_t ev_end[n_kernels];
  for (int i = 0; i < n_kernels; i++) {
    checkCudaErrors(cudaEventCreate(&ev_start[i]));
    checkCudaErrors(cudaEventCreate(&ev_end[i]));
  }
  
  for (int i = 0; i < 2; i++) {
    checkCudaErrors(cudaStreamSynchronize(streams[i]));
  }

  // Start low priority kernel
  checkCudaErrors(
      cudaEventRecord(ev_start[0], streams[low_priority_stream_idx]));
  low_preempt<<<TBLOCKS, THREADS, 0, streams[0]>>>(d_dst[0], d_src[0],
      size);
  checkCudaErrors(cudaEventRecord(ev_end[0], streams[low_priority_stream_idx]));

  // synchronize on the start, so we launch this after the low priority kernel has started
  checkCudaErrors(cudaEventSynchronize(ev_start[0]));

  // Launch n_kernels - 1 high priority kernels synchronously 
  for (int i = 1; i < n_kernels; i++) {
    checkCudaErrors(
        cudaEventRecord(ev_start[i], streams[high_priority_stream_idx]));
    high_preempt<<<TBLOCKS, THREADS, 0, streams[high_priority_stream_idx]>>>(d_dst[i], d_src[i],
        size);
    checkCudaErrors(
        cudaEventRecord(ev_end[i], streams[high_priority_stream_idx]));
    checkCudaErrors(cudaEventSynchronize(ev_end[i]));
  }

  // wait for the low priority kernel to finish
  checkCudaErrors(cudaEventSynchronize(ev_end[0]));

  // Copy result to host
  checkCudaErrors(
      cudaMemcpyAsync(h_dst[0], d_dst[0], size, cudaMemcpyDeviceToHost, streams[low_priority_stream_idx]));

  for (int i = 1; i < n_kernels; i++) {
    checkCudaErrors(
        cudaMemcpyAsync(h_dst[i], d_dst[i], size, cudaMemcpyDeviceToHost, streams[high_priority_stream_idx]));
  }

//  // check results of kernels
//  for (int i = 0; i < n_kernels; i++) {
//    ERR_NE(memcmp(h_dst[i], h_src[i], size), 0);
//  }

  // Clean up
  for (int i = 0; i < n_kernels; i++) {
    checkCudaErrors(cudaFree(d_src[i]));
    checkCudaErrors(cudaFree(d_dst[i]));
  }
  for (int i = 0; i < n_kernels; i++) {
    free(h_src[i]);
  }
  return cudaSuccess;
}

/**
 * Creates streams with priority ranging from high to low and stores them in the streams array.
 * Streams are ordered from highest to lowest priority. 
 */
cudaError_t createStreams(cudaStream_t* streams, int priority_low,
    int priority_hi, size_t n_streams) {
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking,
            priority_hi + i));
  }
  return cudaSuccess;
}

/**
 * Creates a stream with low priority and starts a long-running kernel on it.
 * Creates a stream with high priority and runs a short-running kernel on it,
 * after the low-priority kernel has begun.
 * -- If preemption works, the run time of the low priority kernel should
 *    be extended by the runtime of the high priority kernel which preempts it.
 */
cudaError_t experiment(int priority_low, int priority_hi) {
  // Create streams
  size_t n_streams = (priority_low - priority_hi) + 1;
  cudaStream_t streams[n_streams];
  checkCudaErrors(createStreams(streams, priority_low, priority_hi, n_streams));
  size_t size = TOTAL_SIZE; // Size of host data
  size_t n_kernels = N_HIGH_KERNELS + 1; // 1 low and N high

  for (int i = 0; i < N_ITERATIONS; i++) {
    checkCudaErrors(solo_test(streams, 0, 1, n_streams, size));
    checkCudaErrors(concurrent_test(streams, 0, 1, n_kernels, size));
  }
  return cudaSuccess;
}
