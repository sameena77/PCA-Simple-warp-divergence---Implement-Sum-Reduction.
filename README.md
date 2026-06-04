## NAME: SAMEENA J
## REG NO: 2305002019

# PCA-Simple-warp-divergence---Implement-Sum-Reduction.
Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.

## Aim:
To implement Sum Reduction using the reduceUnrolling16 kernel and compare its performance with the reduceUnrolling8 kernel using CUDA. Performance is analyzed using execution time and profiling metrics obtained through nvprof.

## Procedure:
1.Create a CUDA program and define the block size as 512 threads.

2.Implement the reduceUnrolling8 and reduceUnrolling16 kernels for parallel sum reduction.

3.Allocate host and device memory and initialize the input data.

4.Compute the reference sum on the CPU and copy the input data to the GPU.

5.Execute both kernels separately and measure their execution times using CUDA events.

6.Copy the partial sums back to the host and verify the correctness of the results.

7.Compare the performance of both kernels using execution time and nvprof profiling metrics. 
## PROGRAM:
```
%%writefile reduction_unroll8_16_simple.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                    \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                    \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                         \
    }                                                                    \
}

#define BDIM 512

// ----------- Unroll-8 kernel -----------
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
//unrolling 8
//Type your code here
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

// convert global data pointer to local block pointer
int *idata = g_idata + blockIdx.x * blockDim.x * 8;

// unrolling 8
if (idx + 7 * blockDim.x < n)
{
    g_idata[idx] += g_idata[idx + blockDim.x]
                  + g_idata[idx + 2 * blockDim.x]
                  + g_idata[idx + 3 * blockDim.x]
                  + g_idata[idx + 4 * blockDim.x]
                  + g_idata[idx + 5 * blockDim.x]
                  + g_idata[idx + 6 * blockDim.x]
                  + g_idata[idx + 7 * blockDim.x];
}

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// ----------- Unroll-16 kernel -----------

__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n)
{


    
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

// convert global data pointer to local block pointer
int *idata = g_idata + blockIdx.x * blockDim.x * 16;

// unrolling 16
if (idx + 15 * blockDim.x < n)
{
    g_idata[idx] += g_idata[idx + blockDim.x]
                  + g_idata[idx + 2 * blockDim.x]
                  + g_idata[idx + 3 * blockDim.x]
                  + g_idata[idx + 4 * blockDim.x]
                  + g_idata[idx + 5 * blockDim.x]
                  + g_idata[idx + 6 * blockDim.x]
                  + g_idata[idx + 7 * blockDim.x]
                  + g_idata[idx + 8 * blockDim.x]
                  + g_idata[idx + 9 * blockDim.x]
                  + g_idata[idx + 10 * blockDim.x]
                  + g_idata[idx + 11 * blockDim.x]
                  + g_idata[idx + 12 * blockDim.x]
                  + g_idata[idx + 13 * blockDim.x]
                  + g_idata[idx + 14 * blockDim.x]
                  + g_idata[idx + 15 * blockDim.x];
}
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result of this block
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// ----------- CPU reference reduction -----------
int cpuReduce(int *h_idata, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += h_idata[i];
    return sum;
}

int main() {
    int size = 1 << 24;  // 16M elements
    int bytes = size * sizeof(int);

    // Allocate and initialize host array
    int *h_idata = (int *)malloc(bytes);
    for (int i = 0; i < size; i++)
        h_idata[i] = rand() & 0xFF;

    int *d_idata, *d_odata;
    CHECK(cudaMalloc((void **)&d_idata, bytes));

    // Grid sizes for unroll-8 and unroll-16
    int grid8  = (size + BDIM * 8 - 1) / (BDIM * 8);
    int grid16 = (size + BDIM * 16 - 1) / (BDIM * 16);
    CHECK(cudaMalloc((void **)&d_odata, grid8 * sizeof(int))); // max needed

    // ---------------- CPU reduction ----------------
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    int cpu_sum = cpuReduce(h_idata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float cpuTime;
    CHECK(cudaEventElapsedTime(&cpuTime, start, stop));

    // ---------------- GPU Unroll-8 ----------------
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceUnrolling8<<<grid8, BDIM>>>(d_idata, d_odata, size);
    cudaError_t err8 = cudaGetLastError();
    if (err8 != cudaSuccess) {
        printf("Kernel launch error (Unroll-8): %s\n", cudaGetErrorString(err8));
        return -1;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float gpuTime8;
    CHECK(cudaEventElapsedTime(&gpuTime8, start, stop));

    int *h_odata = (int *)malloc(grid8 * sizeof(int));
    CHECK(cudaMemcpy(h_odata, d_odata, grid8 * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum8 = 0;
    for (int i = 0; i < grid8; i++) gpu_sum8 += h_odata[i];

    // ---------------- GPU Unroll-16 ----------------
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceUnrolling16<<<grid16, BDIM>>>(d_idata, d_odata, size);
    cudaError_t err16 = cudaGetLastError();
    if (err16 != cudaSuccess) {
        printf("Kernel launch error (Unroll-16): %s\n", cudaGetErrorString(err16));
        return -1;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float gpuTime16;
    CHECK(cudaEventElapsedTime(&gpuTime16, start, stop));

    CHECK(cudaMemcpy(h_odata, d_odata, grid16 * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum16 = 0;
    for (int i = 0; i < grid16; i++) gpu_sum16 += h_odata[i];

    // ---------------- Print results ----------------
    printf("CPU sum   : %d, time %.5f sec\n", cpu_sum, cpuTime / 1000.0f);
    printf("GPU sum-8 : %d, time %.5f ms\n", gpu_sum8, gpuTime8);
    printf("GPU sum-16: %d, time %.5f ms\n", gpu_sum16, gpuTime16);

    // ---------------- Cleanup ----------------
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## Output:
<img width="1279" height="645" alt="image" src="https://github.com/user-attachments/assets/df52a2af-8193-4811-988b-1e8c66fb0366" />


## Result:
The reduceUnrolling16 kernel was successfully implemented and tested. Both kernels produced the same reduction result (2139353471). The execution time of reduceUnrolling16 (0.29488 ms) was lower than reduceUnrolling8 (0.47443 ms), showing improved performance due to increased loop unrolling and reduced kernel overhead
