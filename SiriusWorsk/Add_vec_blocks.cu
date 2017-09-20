#include <stdio.h>
#include <malloc.h>
#include <cuda.h>

#define SIZE 65535

__global__ void VectorAdd(int *a, int *b, int *c)
{
	int i = blockIdx.x;
	if(i<SIZE)
		c[i]=a[i]+b[i];
}


int main()
{

	clock_t start = clock();

	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	
	a = (int *)malloc(SIZE*sizeof(int));
	b = (int *)malloc(SIZE*sizeof(int));
	c = (int *)malloc(SIZE*sizeof(int));
	
	cudaMalloc(&d_a, SIZE*sizeof(int));
	cudaMalloc(&d_b, SIZE*sizeof(int));
	cudaMalloc(&d_c, SIZE*sizeof(int));

	for(int i=0;i<SIZE;i++)
	{
		a[i]=i;
		b[i]=i;
		c[i]=0;	
	}

	cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice);

	
	dim3 dimblock(SIZE,1,1);
	dim3 dimGrid(1,1,1);
	
	VectorAdd<<<dimblock,dimGrid>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

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




