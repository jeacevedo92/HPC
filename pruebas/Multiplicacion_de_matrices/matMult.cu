#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <iostream>

using namespace std;

#define H 1000
#define W 1000

__global__ void multMatCUDA(int *d_a,int *d_b,int *d_c){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row < H && col < W){
    int result = 0;
    for(int k = 0; k < W; k++){
      result += d_a[row * W + k] * d_b[k * W + col];
    }
    d_c[row * W + col] = result;
  }
}

void multMat(int *h_a, int *h_b, int *h_c){
  
  for(int i = 0; i < H; i++){
    for(int j = 0; j < W; j++){
      int result = 0;
      for(int k = 0; k < W; k++){
        result += h_a[i * W + k] * h_b[k * W + j];
      }
      h_c[i * W + j] = result;
    }
  }
}

bool compareTo(int *h_c,int *h_result){
  for(int i = 0; i < H; i++){
    for(int j = 0; j < W; j++){
      if(h_c[i * W + j] != h_result[i * W + j]){
        return false;
			}       
    }
  }
  return true;
}

void printMatrix(int *result){
	for(int i = 0; i < H; i++){
	  for(int j = 0; j < W; j++){
		cout<<result[i * W + j]<<" ";
	  }
	  cout<<endl;
	}
}

int main(){
  clock_t start, end;
  double cpu_time_used, gpu_time_used;
  float blockSize = 32;
  int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_result;
  
  //Asignar memoria en el host
  h_a = (int*)malloc(sizeof(int)*H*W);
  h_b = (int*)malloc(sizeof(int)*H*W);
  h_c = (int*)malloc(sizeof(int)*H*W);
  h_result = (int*)malloc(sizeof(int)*H*W);
  
  //Inicializar las matrices
  for(int i = 0; i < H; i++){
    for(int j=0; j < W; j++){
      h_a[i*W+j] = i;
      h_b[i*W+j] = i+1;
      h_c[i*W+j] = 0;
      }
  }
  
  start = clock();
  //Llamar funcion que sume dos vectores y retorne el resultado en h_c
  multMat(h_a, h_b, h_c);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);
    
  //Asignacion de memoria en el device
  cudaMalloc(&d_a, sizeof(int)*H*W);
  cudaMalloc(&d_b, sizeof(int)*H*W);
  cudaMalloc(&d_c, sizeof(int)*H*W);
  
  //Copiar los datos del host al device
  cudaMemcpy(d_a, h_a, H*W* sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, H*W* sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(W/blockSize),ceil(H/blockSize),1);
  
  start = clock();
  multMatCUDA<<< dimGrid, dimBlock >>>(d_a, d_b, d_c);
  cudaMemcpy(h_result, d_c, H*W*sizeof(int), cudaMemcpyDeviceToHost);
  end = clock();
  gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);
    
  if(compareTo(h_c, h_result)){
  	printf("Matrices Iguales");
  }
  else{
  	printf("Matrices Diferentes");
  }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_result);  
  return 0;
}