# PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24
<h3>NAME : YOGESH. V</h3>
<h3>REGISTER NO : 212223230250</h3>
<h3>EX. NO : 6 </h3>
<h1> <align=center> MATRIX TRANSPOSITION USING SHARED MEMORY </h3>
  Implement Matrix transposition using GPU Shared memory.</h3>

## AIM:
To perform Matrix Multiplication using Transposition using shared memory.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
 CUDA_SharedMemory_AccessPatterns:

1. Begin Device Setup
    1.1 Select the device to be used for computation
    1.2 Retrieve the properties of the selected device
2. End Device Setup

3. Begin Array Size Setup
    3.1 Set the size of the array to be used in the computation
    3.2 The array size is determined by the block dimensions (BDIMX and BDIMY)
4. End Array Size Setup

5. Begin Execution Configuration
    5.1 Set up the execution configuration with a grid and block dimensions
    5.2 In this case, a single block grid is used
6. End Execution Configuration

7. Begin Memory Allocation
    7.1 Allocate device memory for the output array d_C
    7.2 Allocate a corresponding array gpuRef in the host memory
8. End Memory Allocation

9. Begin Kernel Execution
    9.1 Launch several kernel functions with different shared memory access patterns (Use any two patterns)
        9.1.1 setRowReadRow: Each thread writes to and reads from its row in shared memory
        9.1.2 setColReadCol: Each thread writes to and reads from its column in shared memory
        9.1.3 setColReadCol2: Similar to setColReadCol, but with transposed coordinates
        9.1.4 setRowReadCol: Each thread writes to its row and reads from its column in shared memory
        9.1.5 setRowReadColDyn: Similar to setRowReadCol, but with dynamic shared memory allocation
        9.1.6 setRowReadColPad: Similar to setRowReadCol, but with padding to avoid bank conflicts
        9.1.7 setRowReadColDynPad: Similar to setRowReadColPad, but with dynamic shared memory allocation
10. End Kernel Execution

11. Begin Memory Copy
    11.1 After each kernel execution, copy the output array from device memory to host memory
12. End Memory Copy

13. Begin Memory Free
    13.1 Free the device memory and host memory
14. End Memory Free

15. Reset the device

16. End of Algorithm

## PROGRAM:
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
```
%%writefile matrix_mul.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdbool.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif // _COMMON_H

#define SIZE 4
#define BLOCK_SIZE 2

// Host matrix multiplication for verification
void matrixMultiplyHost(int *a, int *b, int *c, int size)
{
    for (int row = 0; row < size; ++row)
    {
        for (int col = 0; col < size; ++col)
        {
            int sum = 0;
            for (int k = 0; k < size; ++k)
            {
                sum += a[row * size + k] * b[k * size + col];
            }
            c[row * size + col] = sum;
        }
    }
}

// GPU kernel
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size)
    {
        int sum = 0;
        for (int k = 0; k < size; ++k)
        {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main()
{
    int a[SIZE * SIZE], b[SIZE * SIZE], c[SIZE * SIZE], c_host[SIZE * SIZE];
    int *dev_a, *dev_b, *dev_c;
    int bytes = SIZE * SIZE * sizeof(int);

    // Initialize matrices
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            a[i * SIZE + j] = i + 1;
            b[i * SIZE + j] = j + 1;
        }
    }

    // Allocate device memory
    CHECK(cudaMalloc((void**)&dev_a, bytes));
    CHECK(cudaMalloc((void**)&dev_b, bytes));
    CHECK(cudaMalloc((void**)&dev_c, bytes));

    // Copy to device
    CHECK(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));

    // CUDA execution
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    gettimeofday(&end, NULL);
    double elapsed_gpu = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    CHECK(cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost));

    // CPU execution
    gettimeofday(&start, NULL);
    matrixMultiplyHost(a, b, c_host, SIZE);
    gettimeofday(&end, NULL);
    double elapsed_cpu = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    // Print results
    printf("Result Matrix from GPU:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c[i * SIZE + j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix from CPU:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c_host[i * SIZE + j]);
        }
        printf("\n");
    }



    printf("GPU Time: %.6f s\n", elapsed_gpu);
    printf("CPU Time: %.6f s\n", elapsed_cpu);

    // Cleanup
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_c));

    return 0;
}
```
```
!nvcc -arch=sm_75 -o matrix_mul matrix_mul.cu
```
```
!./matrix_mul
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/6b967541-ffb1-47fc-be8d-74abaf17e258)


## RESULT:
Thus the program has been executed by using CUDA to transpose a matrix. It is observed that there are variations shared memory and global memory implementation. The elapsed times are recorded as _______________.
