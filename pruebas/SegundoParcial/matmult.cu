#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Multiplicacion de Mini Matriz - Matriz

__global__ void multMatKernel(double *d_a, double *d_b, double *d_c, int NRA,
                              int NCA, int NCB) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < NRA && col < NCB) {
    double result = 0;
    for (int j = 0; j < NCA; j++) {
      result += d_a[row * NCA + j] * d_b[j * NCB + col];
    }
    d_c[row * NCB + col] = result;
  }
}

void multMatCUDA(double *M_a, double *M_b, double *R_c, int NRA, int NCA,
                 int NCB) {
  float blockSize = 32;
  double *d_a, *d_b, *d_c;

  // Asignacion de memoria en el device
  cudaMalloc(&d_a, sizeof(double) * NRA * NCA);
  cudaMalloc(&d_b, sizeof(double) * NCA * NCB);
  cudaMalloc(&d_c, sizeof(double) * NRA * NCB);

  cudaMemcpy(d_a, M_a, NRA * NCA * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, M_b, NCA * NCB * sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(NCB / blockSize), ceil(NRA / blockSize), 1);

  multMatKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, NRA, NCA, NCB);
  cudaMemcpy(R_c, d_c, NRA * NCB * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
