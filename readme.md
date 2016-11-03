# CUDA Preemption Experiments
## Introduction
This set of experiments are designed to test how kernels behave when preempted.
To test this we use CUDA streams with priority and cuda events to synchronize between streams.

## How to use this repository
There are a number of experiments in this repository. Compile all of them at once by running `make` in the root directory. Each experiment can be compiled individually by `cd`ing to the experiment directory and running `make`.

## Experiments
1. one-stream

  We create one low-priority stream and one high-priority stream. The low-priority stream and high-priority stream first run separately and we time them using cuda events. Then, the low-priority stream is launched first and the high-priority stream is launched after, preempting it.

1. n-streams

  We launch n low-priority background kernels and 1 high-priority kernel. This experiment proceeds similarly to the one-stream experiment.

1. priority-interrupt

  We launch one long-running, low-priority kernel and preempt it n consecutive times with a high-priority kernel. We measure the runtime of all kernels.

1. priority-streams

  We create `n` streams and launch `n/m` kernels with each of the `m` stream priority levels. On current architectures `m=2` so this experiment is not that interesting.

## Other files
`common` includes helper functions from the NVIDIA CUDA Samples. 

## Note
To obtain more precise timing measurements, use `nvprof` or the NVIDIA Visual Profiler.

This experiment was created by modifying the NVIDIA CUDA StreamPriorities sample code.
