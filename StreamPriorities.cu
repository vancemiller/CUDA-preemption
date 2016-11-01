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

#include <helper_cuda.h>

#define TOTAL_SIZE  256*1024*1024
#define EACH_SIZE   128*1024*1024

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
__global__ void memcpy_kernel(int *dst, int *src, size_t n)
{
  int num = gridDim.x * blockDim.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = id; i < n / sizeof(int); i += num)
  {
    dst[i] = src[i];
  }
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (int i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

// Forward declaration
int create_streams(int, int, int);

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

  for (int i = 2; i < 12; i += 2) {
    printf("==== %d streams ====\n", i);
    // empirical testing shows that we can create 12 threads on the GTX 1080
    // before this benchmark runsout of memory.
    create_streams(priority_low, priority_hi, i);
  }
  exit(EXIT_SUCCESS);
}

int create_streams(int priority_low, int priority_hi, int n_streams) {
  // create streams with all available priorities
  cudaStream_t streams[n_streams];
  int priority_space = abs(priority_low - priority_hi) + 1;
  for (int i = 0; i < priority_space; i++) {
    for (int j = n_streams / priority_space * i;
        j < n_streams / priority_space * (i + 1); j++) {
      printf("creating stream %d with priority %d\n", j, priority_low - i);
      checkCudaErrors(cudaStreamCreateWithPriority(&streams[j],
          cudaStreamNonBlocking, priority_low - i));
    }
  }

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

  // create some events
  cudaEvent_t ev_start[n_streams];
  cudaEvent_t ev_end[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaEventCreate(&ev_start[i]));
    checkCudaErrors(cudaEventCreate(&ev_end[i]));
  }

  /* */

  // call pair of kernels repeatedly (with different priority streams)
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaEventRecord(ev_start[i], streams[i]));
  }

  for (int i = 0; i < TOTAL_SIZE; i += EACH_SIZE) {
    int j = i / sizeof(int);
    for (int k = 0; k < n_streams; k++) {
      memcpy_kernel<<<TBLOCKS, THREADS, 0, streams[k]>>>(d_dst[k] + j, d_src[k] + j, EACH_SIZE);
    }
  }

  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaEventRecord(ev_end[i], streams[i]));
  }

  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaEventSynchronize(ev_end[i]));
  }

  /* */

  size = TOTAL_SIZE;
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaMemcpy(h_dst[i], d_dst[i], size, cudaMemcpyDeviceToHost));
  }

  // check results of kernels
  for (int i = 0; i < n_streams; i++) {
    ERR_NE(memcmp(h_dst[i], h_src[i], size), 0);
  }

  // check timings
  float ms[n_streams];
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaEventElapsedTime(&ms[i], ev_start[i], ev_end[i]));
  }

  for (int i = 0; i < priority_space; i++) {
    for (int j = n_streams / priority_space * i; j < n_streams / priority_space * (i + 1); j++) {
      printf("elapsed time of kernels launched to %d priority stream: %.3lf ms\n", priority_low - i, ms[j]);
    }
  }

  // Clean up
  for (int i = 0; i < n_streams; i++) {
    checkCudaErrors(cudaFree(d_src[i]));
    checkCudaErrors(cudaFree(d_dst[i]));
  }

  return 0;
}
