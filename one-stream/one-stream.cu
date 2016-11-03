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

// copy from source -> destination arrays
__global__ void slow_kernel(int *dst, int *src, size_t n)
{
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  int ret = 0; // accumulate garbage computation in ret and write it to dest[i]
  for (int i = id; i < n / sizeof(int); i += num) {
    for (int j = 0; j < 1000; j++) {
      ret += src[i] * j;
    }
    dst[i] = ret;
  }
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

// Forward declaration
int preempt_stream(int, int);

int main(int argc, char **argv)
{
  cudaDeviceProp device_prop;
  int dev_id;

  printf("Starting [%s]...\n", argv[0]);

  // set device
  dev_id = findCudaDevice(argc, (const char **) argv);
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

  if ((device_prop.major << 4) + device_prop.minor < 0x35)
  {
    fprintf(stderr, "%s requires Compute Capability of SM 3.5 or higher to run.\nexiting...\n", argv[0]);
    exit(EXIT_WAIVED);
  }

  // get the range of priorities available
  // [ greatest_priority, lowest_priority ]
  int priority_low;
  int priority_hi;
  checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_hi));
  printf("CUDA stream priority range: LOW: %d to HIGH: %d\n", priority_low, priority_hi);

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
  checkCudaErrors(cudaStreamCreateWithPriority(&streams[0],
        cudaStreamNonBlocking, priority_low));
  checkCudaErrors(cudaStreamCreateWithPriority(&streams[1],
        cudaStreamNonBlocking, priority_hi));

  size_t size;
  size = TOTAL_SIZE;

  // initialise host data
  int *h_src[n_streams];
  for (int i = 0; i < n_streams; i++) {
    ERR_EQ(h_src[i] = (int *) malloc(size), NULL);
    mem_init(h_src[i], size);
  }

  // initialise device data
  int *h_dst[n_streams];
  for (int i = 0; i < n_streams; i++) {
    ERR_EQ(h_dst[i] = (int *) malloc(size), NULL);
    memset(h_dst[i], 0, size);
  }

  // copy source data -> device
  int *d_src[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMalloc(&d_src[i], size));
    checkCudaErrors(cudaMemcpy(d_src[i], h_src[i], size, cudaMemcpyHostToDevice));
  }

  // allocate memory for memcopy destination
  int *d_dst[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMalloc(&d_dst[i], size));
  }

  /* */

  // Begin profilling
  checkCudaErrors(cudaProfilerStart());

  // Time low priority on its own
  {
    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventRecord(start, streams[0]));
    slow_kernel<<<TBLOCKS, THREADS, 0, streams[0]>>>(d_dst[0], d_src[0], TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(end, streams[0]));
    checkCudaErrors(cudaEventSynchronize(end));

    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    printf("Low priority solo elapsed time %0.6f ms\n", ms);
  }

  // Time high priority on its own
  {
    cudaEvent_t start, end;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventRecord(start, streams[1]));
    slow_kernel<<<TBLOCKS, THREADS, 0, streams[1]>>>(d_dst[1], d_src[1], TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(end, streams[1]));
    checkCudaErrors(cudaEventSynchronize(end));

    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
    printf("High priority solo elapsed time %0.6f ms\n", ms);
  }

  // Start low priority then interrupt it with high priority
  {
    // create some events
    cudaEvent_t ev_start[n_streams];
    cudaEvent_t ev_end[n_streams];
    for (int i = 0; i < n_streams; i++) {
      checkCudaErrors(cudaEventCreate(&ev_start[i]));
      checkCudaErrors(cudaEventCreate(&ev_end[i]));
    }
  
    checkCudaErrors(cudaEventRecord(ev_start[0], streams[0]));
    slow_kernel<<<TBLOCKS, THREADS, 0, streams[0]>>>(d_dst[0], d_src[0], TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(ev_end[0], streams[0]));

    // synchronize on the start, so we launch this after the low priority kernel has started

    checkCudaErrors(cudaEventSynchronize(ev_start[0]));

    checkCudaErrors(cudaEventRecord(ev_start[1], streams[1]));
    slow_kernel<<<TBLOCKS, THREADS, 0, streams[1]>>>(d_dst[1], d_src[1], TOTAL_SIZE);
    checkCudaErrors(cudaEventRecord(ev_end[1], streams[1]));
    
    checkCudaErrors(cudaEventSynchronize(ev_end[1]));
    checkCudaErrors(cudaEventSynchronize(ev_end[0]));

    float ms[n_streams];
    for (int i = 0; i < n_streams; i++) {
      checkCudaErrors(cudaEventElapsedTime(&ms[i], ev_start[i], ev_end[i]));
    }
    printf("Low priority preempted by high priority test\n");
    printf("Low priority elapsed time %0.6f ms\n", ms[0]);
    printf("High priority elapsed time %0.6f ms\n", ms[1]);

  }

  // Stop profiling
  checkCudaErrors(cudaProfilerStop());

  /* */

  size = TOTAL_SIZE;
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMemcpy(h_dst[i], d_dst[i], size, cudaMemcpyDeviceToHost));
  }

  // check results of kernels
  /*
  // If we were doing some easily checkable computation, we could 
  // verify that the result is correct here
  for (int i = 0; i < n_streams; i++) {
    ERR_NE(memcmp(h_dst[i], h_src[i], size), 0);
  }
  */

  // Clean up
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaFree(d_src[i]));
    checkCudaErrors(cudaFree(d_dst[i]));
  }

  return 0;
}
