#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width){
   
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   for (int j = 0; j < Mask_Width; j++) {
       if(N_start_point +j >= 0 && N_start_point + j < Width){
           Pvalue += N[N_start_point+j] * M[j];
       }
   }
   P[i] = Pvalue;
}

void convolution_1D_basic(float *N, float *M, float *P, int Mask_Width, int Width){
   for (int i = 0; i < Width; i++) {
       float Pvalue = 0;
       int N_start_point = i - (Mask_Width/2);
       for (int j = 0; j < Mask_Width; j++) {
           if(N_start_point +j >= 0 && N_start_point + j < Width){
               Pvalue += N[N_start_point+j] * M[j];
           }
       }
       P[i] = Pvalue;
   }
   
}

bool compareTo(float *P,float *P_result, int W){
  bool flag = true;
  for(int i = 0; i < W; i++){
    if(P[i] != P_result[i]){
    	flag = false;
      break;
    }
  }
  return flag;
}

int main(){
    clock_t start, end;
    double cpu_time_used, gpu_time_used;
    float *N, *M, *P, *P_result, *d_N, *d_M, *d_P;
    int Width = 7, Mask_Width = 5;
    //Asignar memoria en el host
    N = (float*)malloc(Width*sizeof(float));
    M = (float*)malloc(Mask_Width*sizeof(float));
    P = (float*)malloc(Width*sizeof(float));
    P_result = (float*)malloc(Width*sizeof(float));
    
    //Inicializar los valores
    for (int i = 0; i < Width; i++) {
        N[i] = i + 1;
        P[i] = 0;
        P_result[i] = 0;
    }
    
    M[0] = 3;
    M[1] = 4;
    M[2] = 5;
    M[3] = 4;
    M[4] = 3;
    
    start = clock();
    //Llamar funcion hace la convolucion de vectores en el host y retorne el resultado en P
    convolution_1D_basic(N, M, P, Mask_Width, Width);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);
    
    // //Asignacion de memoria en el device
    cudaMalloc(&d_N, Width*sizeof(float));
    cudaMalloc(&d_M, Mask_Width*sizeof(float));
    cudaMalloc(&d_P, Width*sizeof(float));
    
    //Copiar los datos del host al device
    cudaMemcpy(d_N, N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, Width * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(4, 1, 1);
    dim3 dimGrid((Width / dimBlock.x) + 1, 1, 1);
    
    start = clock();
    //Llamar funcion hace la convolucion de vectores en el device y retorne el resultado en P_result
    convolution_1D_basic_kernel<<< dimGrid, dimBlock >>>(d_N, d_M, d_P, Mask_Width, Width);
    cudaMemcpy(P_result, d_P, Width*sizeof(float), cudaMemcpyDeviceToHost);
    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);
    
    if(compareTo(P, P_result, Width)){
        cout << "Vectores Iguales" << endl;
    }
    else{
        cout <<"Vectores Diferentes" << endl;
    }
    // for (int i = 0; i < Width; i++) {
    //     cout << P_result[i] <<endl;
    // }
    
    free(N);
    free(M);
    free(P);
    free(P_result);
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    
    return 0;
}