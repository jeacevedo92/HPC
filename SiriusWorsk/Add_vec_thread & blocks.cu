#include <stdio.h>
#include <malloc.h>
#include <cuda.h>

#define SIZE_thread 1024

__global__ void VectorAdd(int *A, int *B, int *C,int n)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<n)
		C[i]=A[i]+B[i];
}


int main()
{

	int n = 3000;

	clock_t start = clock();

	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	
	a = (int *)malloc(n*sizeof(int));
	b = (int *)malloc(n*sizeof(int));
	c = (int *)malloc(n*sizeof(int));
	
	cudaMalloc(&d_a, n*sizeof(int));
	cudaMalloc(&d_b, n*sizeof(int));
	cudaMalloc(&d_c, n*sizeof(int));

	for(int i=0;i<n;i++)
	{
		a[i]=i;
		b[i]=i;
		c[i]=0;	
	}

	cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, n*sizeof(int), cudaMemcpyHostToDevice);

	
	dim3 dimGrid(ceil(n/float(SIZE_thread)),1,1);
	dim3 dimblock(SIZE_thread,1,1);

	VectorAdd<<<dimGrid,dimblock>>>(d_a, d_b, d_c,n);

	cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0;i<10; i++)
		printf("%d ",c[i]);
		
	free(a);
	free(b);
	free(c);
	
	cudaFree(d_a);	
	cudaFree(d_b);
	cudaFree(d_c);
	
	printf("Tiempo transcurrido: %f \n ",((double)clock() - start) / CLOCKS_PER_SEC);

	return 0;
}




