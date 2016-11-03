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

#define TOTAL_SIZE  16 * 1024*1024

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
#define N_ITERATIONS 1024
#define N_HIGH_KERNELS 1

// copy from source -> destination arrays
__device__ void slow_kernel(int *dst, int *src, size_t n) {
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = id; i < n / sizeof(int); i += num) {
#define DELAY 1024
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
int preempt_stream(int, int);

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
    exit(EXIT_WAIVED);
  }

  // get the range of priorities available
  // [ greatest_priority, lowest_priority ]
  int priority_low;
  int priority_hi;
  checkCudaErrors(
      cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));
  printf("CUDA stream priority range: LOW: %d to HIGH: %d\n", priority_low,
      priority_hi);

  preempt_stream(priority_low, priority_hi);

  exit(EXIT_SUCCESS);
}

/**
 * Creates a stream with low priority and starts a long-running kernel on it.
 * Creates a stream with high priority and runs a short-running kernel on it,
 * after the low-priority kernel has begun.
 * -- If preemption works, the run time of the low priority kernel should
 *    be extended by the runtime of the high priority kernel which preempts it.
 */
int preempt_stream(int priority_low, int priority_hi) {
  // Create streams
  size_t n_streams = 2; // Two streams (low and high)
                        // let index 0 hold low and 1 hold high
  cudaStream_t streams[n_streams];
  checkCudaErrors(
      cudaStreamCreateWithPriority(&streams[0], cudaStreamNonBlocking,
          priority_low));
  checkCudaErrors(
      cudaStreamCreateWithPriority(&streams[1], cudaStreamNonBlocking,
          priority_hi));

  size_t size;
  size = TOTAL_SIZE;

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
        cudaMemcpy(d_src[i], h_src[i], size, cudaMemcpyHostToDevice));
  }

  // allocate memory for memcopy destination
  int *d_dst[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMalloc(&d_dst[i], size));
  }

  /* */

  // Begin profiling
  checkCudaErrors(cudaProfilerStart());

  // Time low priority on its own
  double low_ms;
  for (int iteration = 0; iteration < N_ITERATIONS; iteration++) {
    float ms;
    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventRecord(start, streams[0]));
    low<<<TBLOCKS, THREADS, 0, streams[0]>>>(d_dst[0], d_src[0], TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(end, streams[0]));
    checkCudaErrors(cudaEventSynchronize(end));

    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    low_ms += ((double) ms) / ((double) N_ITERATIONS);
  }
  printf("Low priority solo average elapsed time %0.6f ms\n", low_ms);

  // Time high priority on its own
  double high_ms;
  for (int iteration = 0; iteration < N_ITERATIONS; iteration++) {
    float ms;
    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventRecord(start, streams[1]));
    high<<<TBLOCKS, THREADS, 0, streams[1]>>>(d_dst[1], d_src[1], TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(end, streams[1]));
    checkCudaErrors(cudaEventSynchronize(end));

    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    high_ms += ((double) ms) / ((double) N_ITERATIONS);
  }
  printf("High priority solo average elapsed time %0.6f ms\n", high_ms);

  // Start low priority then interrupt it with high priority
  double low_preempt_ms;
  double high_preempt_ms;

  for (int iteration = 0; iteration < N_ITERATIONS; iteration++) {
    // create some events
    cudaEvent_t ev_start[1 + N_HIGH_KERNELS];
    cudaEvent_t ev_end[1 + N_HIGH_KERNELS];
    for (int i = 0; i < 1 + N_HIGH_KERNELS; i++) {
      checkCudaErrors(cudaEventCreate(&ev_start[i]));
      checkCudaErrors(cudaEventCreate(&ev_end[i]));
    }

    checkCudaErrors(cudaEventRecord(ev_start[0], streams[0]));
    low_preempt<<<TBLOCKS, THREADS, 0, streams[0]>>>(d_dst[0], d_src[0],
        TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(ev_end[0], streams[0]));

    // synchronize on the start, so we launch this after the low priority kernel has started

    checkCudaErrors(cudaEventSynchronize(ev_start[0]));

    for (int j = 0; j < N_HIGH_KERNELS; j++) {
      checkCudaErrors(cudaEventRecord(ev_start[1 + j], streams[1]));
      high_preempt<<<TBLOCKS, THREADS, 0, streams[1]>>>(d_dst[1], d_src[1],
          TOTAL_SIZE);
      checkCudaErrors(cudaEventRecord(ev_end[1 + j], streams[1]));

      checkCudaErrors(cudaEventSynchronize(ev_end[1 + j]));

    }

    checkCudaErrors(cudaEventSynchronize(ev_end[0]));

    float ms[1 + N_HIGH_KERNELS];
    for (int i = 0; i < 1 + N_HIGH_KERNELS; i++) {
      checkCudaErrors(cudaEventElapsedTime(&ms[i], ev_start[i], ev_end[i]));
    }
    low_preempt_ms += ((double) ms[0]) / ((double) N_ITERATIONS);
    for (int i = 0; i < N_HIGH_KERNELS; i++) {
      high_preempt_ms += ((double) ms[1 + i]) / ((double) N_ITERATIONS) / ((double) N_HIGH_KERNELS);
    }

  }
  printf("Low priority preempted by high priority test\n");
  printf("Low priority elapsed time %0.6f ms\n", low_preempt_ms);
  printf("High priority elapsed time %0.6f ms\n", high_preempt_ms);
  // these numbers aren't reliable. use nvprof to get better times
  printf("Overhead of context switch %0.6f ms\n",
      (low_preempt_ms - low_ms - high_ms) + (high_preempt_ms - high_ms));

  // Stop profiling
  checkCudaErrors(cudaProfilerStop());

  /* */

  size = TOTAL_SIZE;
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(
        cudaMemcpy(h_dst[i], d_dst[i], size, cudaMemcpyDeviceToHost));
  }

  // check results of kernels
  for (int i = 0; i < n_streams; i++) {
    ERR_NE(memcmp(h_dst[i], h_src[i], size), 0);
  }

  // Clean up
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaFree(d_src[i]));
    checkCudaErrors(cudaFree(d_dst[i]));
  }

  return 0;
}
