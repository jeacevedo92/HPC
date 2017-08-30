#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define N 128

__global__ void kernel_add(int *d_a,int *d_b,int *d_c){
  //int i = threadIdx.x;
  //int i = blockIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N)
    d_c[i] = d_a[i] + d_b[i];
}

void addVector(int *h_a,int *h_b,int *h_c){
  for(int i = 0; i < N; i++){
  	h_c[i] = h_a[i] + h_b[i];
  }
}

bool compareTo(int *h_c,int *h_result){
  bool flag = true;
  for(int i = 0; i < N; i++){
    if(h_c[i] != h_result[i]){
    	flag = false;
      break;
    }
  }
  return flag;
}

int main(){
  clock_t start, end;
  double cpu_time_used, gpu_time_used;
	int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_result;
  
  //Asignar memoria en el host
  h_a = (int*)malloc(N*sizeof(int));
  h_b = (int*)malloc(N*sizeof(int));
  h_c = (int*)malloc(N*sizeof(int));
  h_result = (int*)malloc(N*sizeof(int));
  
  //Inicializar los vectores
  for(int i = 0; i < N; i++){
  	h_a[i] = i;
    h_b[i] = i+1;
    h_c[i] = 0;
  }
  
  start = clock();
  //Llamar funcion que sume dos vectores y retorne el resultado en h_c
  addVector(h_a, h_b, h_c);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);
    
  //Asignacion de memoria en el device
  cudaMalloc(&d_a, N*sizeof(int));
  cudaMalloc(&d_b, N*sizeof(int));
  cudaMalloc(&d_c, N*sizeof(int));
  
  //Copiar los datos del host al device
  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, N * sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 threads_per_block(10, 1, 1);
  dim3 number_of_blocks((N / threads_per_block.x) + 1, 1, 1);
  
  start = clock();
  //Lanzar el kernel
  //kernel_add<<<1, N>>>(d_a, d_b, d_c);  
  //kernel_add<<<N, 1>>>(d_a, d_b, d_c);
  kernel_add<<< number_of_blocks, threads_per_block >>>(d_a, d_b, d_c);
  cudaMemcpy(h_result, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
  end = clock();
  gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);
  
  if(compareTo(h_c, h_result)){
  	printf("Vectores Iguales");
  }
  else{
  	printf("Vectores Diferentes");
  }
  return 0;

}
