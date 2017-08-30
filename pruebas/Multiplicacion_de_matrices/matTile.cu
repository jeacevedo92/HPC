#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

#define MRows 5
#define MCols 5
#define NRows 5
#define NCols 6
#define PRows 5
#define PCols 6
#define H 10
#define W 10
#define TILE_WIDTH 2

__global__ void MultTiled(float *M, float *N, float *P) {
  __shared__ int ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ int ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float PValue = 0.0;
  for (int p = 0; p < (TILE_WIDTH + MCols - 1) / TILE_WIDTH; p++) {

    if (p * TILE_WIDTH + tx < MCols && Row < MRows)
      ds_M[ty][tx] = M[Row * MCols + p * TILE_WIDTH + tx];
    else
      ds_M[ty][tx] = 0.0;

    if (p * TILE_WIDTH + ty < NRows && Col < NCols)
      ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * NCols + Col];
    else
      ds_N[ty][tx] = 0.0;

    __syncthreads();

    for (int n = 0; n < TILE_WIDTH; ++n)
      PValue += ds_M[ty][n] * ds_N[n][tx];

    __syncthreads();
  }

  if (Row < PRows && Col < PCols)
    P[(Row * PCols) + Col] = PValue;
}

void printMatrix(float *result, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      cout << result[i * C + j] << " ";
    }
    cout << endl;
  }
}

int main() {
  clock_t start, end;
  double gpu_time_used, tiles_time_used;
  float blockSize = 2;
  float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_result, *M, *N, *P, *t_result;

  // Asignar memoria en el host
  h_a = (float *)malloc(sizeof(float) * MRows * MCols);
  h_b = (float *)malloc(sizeof(float) * NRows * NCols);
  h_c = (float *)malloc(sizeof(float) * PRows * PCols);
  h_result = (float *)malloc(sizeof(float) * PRows * PCols);
  t_result = (float *)malloc(sizeof(float) * PRows * PCols);

  // Inicializar las matrices
  for (int i = 0; i < MRows; i++) {
    for (int j = 0; j < MCols; j++) {
      h_a[i * MCols + j] = 1.0;
    }
  }
  cout << "  M1  " << endl;
  printMatrix(h_a, MRows, MCols);

  for (int i = 0; i < NRows; i++) {
    for (int j = 0; j < NCols; j++) {
      h_b[i * NCols + j] = 1.0;
    }
  }

  cout << "  M2  " << endl;
  printMatrix(h_b, NRows, NCols);
  // Asignacion de memoria en el device
  // cudaMalloc(&d_a, sizeof(int) * H * W);
  // cudaMalloc(&d_b, sizeof(int) * H * W);
  // cudaMalloc(&d_c, sizeof(int) * H * W);
  cudaMalloc(&M, sizeof(float) * MRows * MCols);
  cudaMalloc(&N, sizeof(float) * NRows * NCols);
  cudaMalloc(&P, sizeof(float) * PRows * PCols);

  // Copiar los datos del host al device
  // cudaMemcpy(d_a, h_a, H * W * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_b, h_b, H * W * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(M, h_a, MRows * MCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(N, h_b, NRows * NCols * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(NCols / blockSize), ceil(MRows / blockSize), 1);

  // start = clock();
  // multMatCUDA<<< dimBlock, dimGrid >>>(d_a, d_b, d_c);
  // cudaMemcpy(h_result, d_c, H*W*sizeof(int), cudaMemcpyDeviceToHost);
  // end = clock();
  // gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  // printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);

  start = clock();
  MultTiled<<<dimBlock, dimGrid>>>(M, N, P);
  cudaDeviceSynchronize();
  cudaMemcpy(t_result, P, PRows * PCols * sizeof(float),
             cudaMemcpyDeviceToHost);
  end = clock();
  tiles_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU = %lf s\n", tiles_time_used);

  printMatrix(t_result, PRows, PCols);
  // cout<<"tiles :"<<endl;
  // printMatrix(t_result);
  // if (compareTo(t_result, h_result)) {
  //   printf("Matrices Iguales");
  // } else {
  //   printf("Matrices Diferentes");
  // }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(M);
  cudaFree(N);
  cudaFree(P);
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_result);
  free(t_result);
  return 0;
}
