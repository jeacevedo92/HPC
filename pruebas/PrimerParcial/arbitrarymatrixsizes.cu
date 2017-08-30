#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

#define MRows 1600
#define MCols 1600
#define NRows 1600
#define NCols 1500
#define PRows 1600
#define PCols 1500
#define H 10
#define W 10
#define TILE_WIDTH 32


// Mulitiplicacion de matrices en paralelo con TILES
__global__ void MultTiled(float *M, float *N, float *P) {
  // Definimos los tiles para ambas matrices
  __shared__ int ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ int ds_N[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float PValue = 0.0;
  
  for (int p = 0; p < MCols / TILE_WIDTH; p++) {
    // Verficamos que el elemento este dentro de la matriz M
    if (Row < MRows && (p * TILE_WIDTH + tx) < MCols)
      ds_M[ty][tx] = M[Row * MCols + (p * TILE_WIDTH + tx)];
    else
    // Si no esta dentro de la matriz se asigna un 0
      ds_M[ty][tx] = 0.0;
    // Verficamos que el elemento este dentro de la matriz N
    if (Col < NCols && (p * TILE_WIDTH + ty) < MCols)
      ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * NCols + Col];
    else
    // Si no esta dentro de la matriz se asigna un 0
      ds_N[ty][tx] = 0.0;

    __syncthreads();
    // Realiza la operacion de multiplicacion con los valores que hay dentro del TILE
    for (int n = 0; n < TILE_WIDTH; ++n)
      PValue += ds_M[ty][n] * ds_N[n][tx];

    __syncthreads();
  }
  // Guardamos los valores calculados en la multiplicacion en la matriz de resultados
  if (Row < PRows && Col < PCols)
    P[(Row * PCols) + Col] = PValue;
}

// Multiplicacion de matrices en paralelo
__global__ void multMatCUDA(float *d_a, float *d_b, float *d_c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < PRows && col < PCols) {
    float result = 0;
    for (int k = 0; k < MCols; k++) {
      result += d_a[row * MCols + k] * d_b[k * NCols + col];
    }
    d_c[row * PCols + col] = result;
  }
}

// Multiplicacion de matrices secuencialmente
void multMat(float *h_a, float *h_b, float *h_c){
  for(int i = 0; i < PRows; i++){
    for(int j = 0; j < PCols; j++){
      float result = 0;
      for(int k = 0; k < MCols; k++){
        result += h_a[i * MCols + k] * h_b[k * NCols + j];
      }
      h_c[i * PCols + j] = result;
    }
  }
}

// Compara si dos matrices son iguales
bool compareTo(float *h_c,float *h_result){
  for(int i = 0; i < PRows; i++){
    for(int j = 0; j < PCols; j++){
      if(h_c[i * PCols + j] != h_result[i * PCols + j]){
        return false;
			}       
    }
  }
  return true;
}

// Imprime los valores de una matriz
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
  double gpu_time_used, tiles_time_used, cpu_time_used;
  float blockSize = 32;
  float *h_a, *h_b, *h_c, *h_result, *M, *N, *P, *t_result;

  // Asignar memoria en el host
  h_a = (float *)malloc(sizeof(float) * MRows * MCols);
  h_b = (float *)malloc(sizeof(float) * NRows * NCols);
  h_c = (float *)malloc(sizeof(float) * PRows * PCols);
  h_result = (float *)malloc(sizeof(float) * PRows * PCols);
  t_result = (float *)malloc(sizeof(float) * PRows * PCols);

  // Inicializar la primer matriz
  for (int i = 0; i < MRows; i++) {
    for (int j = 0; j < MCols; j++) {
      h_a[i * MCols + j] = 1.0;
    }
  }
  
  // Inicializar la segunda matriz
  for (int i = 0; i < NRows; i++) {
    for (int j = 0; j < NCols; j++) {
      h_b[i * NCols + j] = 1.0;
    }
  }
  
  // Llamado a la multiplicacion de matrices secuencial
  start = clock();
  multMat(h_a, h_b, h_c);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);
 
  // Asignacion de memoria en el device
  cudaMalloc(&M, sizeof(float) * MRows * MCols);
  cudaMalloc(&N, sizeof(float) * NRows * NCols);
  cudaMalloc(&P, sizeof(float) * PRows * PCols);

  // Copiar los datos del host al device
  cudaMemcpy(M, h_a, MRows * MCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(N, h_b, NRows * NCols * sizeof(float), cudaMemcpyHostToDevice);

  // Se definen el numero de bloques y el numero de hilos por bloque
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(PCols / float(blockSize)), ceil(PRows / float(blockSize)),
               1);

  // Llamado a la multiplicacion de matrices en paralelo
  // start = clock();
  // multMatCUDA<<<dimGrid, dimBlock>>>(M, N, P);
  // cudaMemcpy(h_result, P, PRows * PCols * sizeof(float),
  //           cudaMemcpyDeviceToHost);
  // end = clock();
  // gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  // printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);

  // Llamado a la multiplicacion de matrices en paralelo con TILES
  start = clock();
  MultTiled<<<dimGrid, dimBlock>>>(M, N, P);
  cudaDeviceSynchronize();
  cudaMemcpy(t_result, P, PRows * PCols * sizeof(float),
             cudaMemcpyDeviceToHost);
  end = clock();
  tiles_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU con Tiles = %lf s\n", tiles_time_used);

  // Comparar si las matrices resultantes son iguales
  // printMatrix(h_c, PRows, PCols);
  // printMatrix(t_result, PRows, PCols);
  if (compareTo(h_c, t_result)) {
    printf("Matrices Iguales");
  } else {
    printf("Matrices Diferentes");
  }
  
  // Liberar memoria en el device y en el host
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