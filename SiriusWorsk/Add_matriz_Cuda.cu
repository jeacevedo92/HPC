//PROGRAMA QUE SUMA DOS MATRICES EN C++

#include<malloc.h>
#include<cuda.h>


#define Size 1024

__global__ void SumaMatricesCU(int* A,int* B,int* C,int width){
	int col=blockIdx.x*blockDim.x + threadIdx.x;//columnas
	int row=blockIdx.y*blockDim.y + threadIdx.y;//filas

	if((row<width)&&(col<width)){
		C[row*width+col] = A[row*width+col]+B[row*width+col];
	}	
}


void imprimeMatriz(int* A, int width){
	for(int i=0;i<width;i++){
		for(int j=0;j<width;j++){
			printf("%d", A[(i*width)+j]);
		}
		printf("\n");
	}

}

void inicializaMatriz(int* X,int width)
{
	for(int i=0; i < width*width ; i++)
	{ 
		X[i]=1;
	}

}


int main()
{


	cudaError_t error = cudaSuccess;

	int *h_A,*h_B,*h_C,*d_A,*d_B,*d_C;
	
	int width = 2048;

	int size = width * width * sizeof(int);




	// reserva de memoria para las matrices en el host

	h_A = (int*)malloc(size);	
	h_B = (int*)malloc(size);	
	h_C = (int*)malloc(size);



	// inicializa matrices
	inicializaMatriz(h_A,width);
	inicializaMatriz(h_B,width);


	// reserva de memoria para las matrices en el device

	error = cudaMalloc((void**)&d_A,size);
	
	if(error != cudaSuccess){
		printf("Error reservando memoria para d_M");
		exit(0);
	}
	
	error = cudaMalloc((void**)&d_B,size);

	if(error != cudaSuccess){
		printf("Error reservando memoria para d_N");
		exit(0);
	}

	error = cudaMalloc((void**)&d_C,size);
	
	if(error != cudaSuccess){
		printf("Error reservando memoria para d_P");
		exit(0);
	}

	
	//copiando del host al device

	error = cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);//destino d_A y origen A
	
	if(error != cudaSuccess){
		printf("Error COPIANDO memoria para d_A");
		exit(0);
	}

	error = cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
	
	if(error != cudaSuccess){
		printf("Error COPIANDO memoria para d_B");
		exit(0);
	}
	error = cudaMemcpy(d_C,h_C,size,cudaMemcpyHostToDevice);
	
	if(error != cudaSuccess){
		printf("Error COPIANDO memoria para d_C");
		exit(0);
	}

	//47*63*1024=3032064  esta es la cantidad de hilos que vamos a utilizar para hacer la suma de las matrices
	//porque las matrices tienen una dimensión de 2000*1500=3000000 
	//32*32 = 1024 hilos en cada bloque
	//2000/32=63, 1500/32=47
	
	dim3 dimblock(32,32,1);//dimensión de los bloques(cantidad de hilos que se van a utilizar)
	dim3 dimGrid(ceil(width/32),ceil(width/32),1);//dimensión de la malla (cantidad de bloques que se van a utilizar)
	
	SumaMatricesCU<<<dimGrid,dimblock>>>(d_A,d_B,d_C,width);
	
	cudaDeviceSynchronize();//espera que termine la funcion anterior 
	
	error = cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);//copia la operacion relizada en el device al host en el vector C
	
	if(error != cudaSuccess){
		printf("Error copiando d to h memoria para d_C");
		exit(0);
	}

	imprimeMatriz(h_C,width);
	
	free(h_A);free(h_B);free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
	
	
	return 0;

}

