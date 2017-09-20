#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<cassert>
#include<stdio.h>

using namespace std;

void llenaMatrices(int** X,int filas, int columnas)
{
	srand(time(NULL));
	for (int i = 0; i < filas; i++){
			for(int j=0;j<columnas;j++){
				X[i][j]=rand()%10;
			}
	}
}

void imprimeMatrices(int** X,int filas,int columnas)
{
	for (int i = 0; i < filas; i++){
			for(int j=0;j<columnas;j++){
				cout<<X[i][j]<<" ";
			}
			cout<<endl;
	}
}

void inicializaMatriz(int** X,int filas, int columnas){//la llena de ceros
	for(int i=0;i<filas;i++){
		for(int j=0;j<columnas;j++){
			X[i][j]=0;
		}
	}
}

void multiplicaMatrices(int** X,int filX,int colX,int** Y,int filY,int colY,int** Z)
{
	int suma=0;
	
	for(int i=0;i<filX;i++){//la multiplicación se realiza entre las filas de la primera matriz
		for(int j=0;j<colY;j++){//y las columnas de la segunda matriz
			suma=0;//reiniciamos una vez puesto el valor correspondiente en la matriz Z
			for(int k=0;k<filY;k++){
				suma=suma+X[i][k]*Y[k][j];
			}
			Z[i][j]=suma;//este es el valor de la multiplicación
		}	
	}
}

int main(){

	
	
	
	int** A;int** B;int** C;//Matrices
	
	int filA,filB,colA,colB;
	
	printf("Ingrese filas de la matriz A: ");
	scanf("%d",&filA);
	
	printf("Ingrese Columnas de la matriz A: ");
	scanf("%d",&colA);
	
	printf("Ingrese filas de la matriz B");
	scanf("%d",&filB);
	
	printf("Ingrese Columnas de la matriz B");
	scanf("%d",&colB);
	
	
	A=(int**)malloc(filA*sizeof(int*));//reservamos memoria

	for(int i=0;i<filA;i++)
	{
		A[i]=(int*)malloc(colA*sizeof(int*));
	}

	B=(int**)malloc(filB*sizeof(int*));//reservamos memoria
	
	for(int i=0;i<filB;i++)
	{
		B[i]=(int*)malloc(colB*sizeof(int*));
	}
	
	C=(int**)malloc(filA*sizeof(int*));//reservamos memoria
	
	for(int i=0;i<filA;i++)
	{
		C[i]=(int*)malloc(colB*sizeof(int*));
	}
	
	llenaMatrices(A,filA,colA);
	cout<<"Matriz A:"<<endl;
	imprimeMatrices(A,filA,colA);

	llenaMatrices(B,filB,colB);
	cout<<"Matriz B:"<<endl;
	imprimeMatrices(B,filB,colB);
	
	if(colA!=filB){
		cout<<"No se pueden multiplicar"<<endl;
		return 0;
	}
		else
		{
			multiplicaMatrices(A,filA,colA,B,filB,colB,C);
			cout<<"Multiplicación:"<<endl;
			imprimeMatrices(C,filA,colB);
		}	
	
	free(*A);free(*B);free(*C);//liberamos memoria
	free(A);free(B);free(C);

	return 0;
	
}
