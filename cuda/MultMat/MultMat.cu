#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <fstream>

using namespace std;

__global__ void matrixMulKernel(int *d_M, int *d_N, int *d_P, int width){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < width)&&(col < width)){
        Pvalue = 0;
        for (int k = 0; k < width ; ++k){
            Pvalue += d_M[row*width+k] * d_N[k*width+col];
        }
        d_P[row*width+col] = Pvalue;
    }
}

int matrixMulHost(int *h_M, int *h_N, int *h_P, int width){
    int Pvalue;

    for(int row = 0; row < width ; ++row){
        for(int col = 0; col < width ; ++col){
            Pvalue = 0;
            for(int k = 0; k < width ; ++k){
                Pvalue += h_M[row*width+k] * h_N[k*width+col];
            }
            h_P[row*width+col] = Pvalue;
        }
    }
    return 0;
}

int initValues(int *data, int width){
    for(int i = 0; i < width*width; i++)
        data[i] = 2;
    return 0;
}

int printData(int *data, int width){
    for(int i = 0; i < width; ++i){
        for(int j = 0; j < width; ++j){
            printf("%d ", data[(i*width)+j]);
        }
        printf("\n");
    }
    return 0;
}


int initValues(int *data, int width){
    for(int i = 0; i < width*width; i++)
        data[i] = rand()%10;
    return 0;
}


int main(int argc, char const *argv[])
{

	//device
	int *d_MA, *d_MB,*d_MR;

	//host matrix A,  matrix B,  matrix result,  matrix result Device,
	int *h_MA, *h_MB, *h_MR,*h_MRD;

	//width of matrix
  int width = 2048;

	cudaError_t error = cudaSuccess;
	int size = width * width * sizeof(int);

	//clock
	clock_t start, end, startGPU, endGPU;
	double cpu_time, gpu_time;

	 // Allocate memory for each matrix on host
	h_MA = (int*)malloc(size);
	h_MB = (int*)malloc(size);
	h_MR = (int*)malloc(size);
	h_MRD = (int*)malloc(size);

	initValues(h_MA, width);
	initValues(h_MB, width);

	/////////Algoritmo Secuencial////////////////////////////////////////////
	start = clock();
	matrixMulHost(h_MA, h_MB, h_MR, width);
	end = clock();
	cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Tiempo algoritmo secuencial: %.10f\n", cpu_time);

	//reservando memoria para el Device//////////////////////////////////////////

	error = cudaMalloc((void**)&d_MA,size);
	if(error != cudaSuccess){
			printf("Error reservando memoria para MAtrix A en device");
			exit(0);
	}

	error = cudaMalloc((void**)&d_MB,size);
	if(error != cudaSuccess){
			printf("Error reservando memoria para MAtrix B en device");
			exit(0);
	}

	error = cudaMalloc((void**)&d_MR,size);
	if(error != cudaSuccess){
			printf("Error reservando memoria para MAtrix resultado en device");
			exit(0);
	}

	//////////////////////Algoritmo Paralelo///////////////////////////
	///////////////// copiando matrices del Host al device////////////
	startGPU = clock();
	error = cudaMemcpy(d_MA, h_MA, size, cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
			printf("Error copiando matriz A del host al Device");
			exit(0);
	}

	error = cudaMemcpy(d_MB, h_MB, size, cudaMemcpyHostToDevice);
	if(error != cudaSuccess){
			printf("Error copiando  matriz B del host al Device");
			exit(0);
	}

	/////////Lanzamiento de kernel///////////////////////////////////

	int blockSize = 32;
	dim3 dimBlock(blockSize,blockSize,1);
	dim3 dimGrid(ceil(width/float(blockSize)),ceil(width/float(blockSize)),1);
	matrixMulKernel<<<dimGrid,dimBlock>>>(d_MA,d_MA,d_MR,width);
	cudaDeviceSynchronize();
	cudaMemcpy(h_MRD, d_MR, size, cudaMemcpyDeviceToHost);
	endGPU = clock();
	gpu_time = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
	printf("Tiempo algoritmo paralelo: %.10f\n", gpu_time_used);
	printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);
	///////////////////////Algoritmo Paralelo////////////////////////////

	free(h_MA);
	free(h_MB);
	free(h_MR);
	cudaFree(d_MA);
	cudaFree(d_MB);
	cudaFree(d_MR);
	return 0;
}
