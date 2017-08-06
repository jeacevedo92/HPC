#include<stdlib.h>
#include<time.h>
#include<stdio.h>


void llenaMatrices(FILE *archivo,int** matriz,int rows, int col){

	int i,j;

	for( i=0 ; i<rows ; i++){
		for(j=0;j<col;j++){
			fgetc(archivo);
			fscanf (archivo,"%d",&matriz[i][j]);
		}
	}
}

void imprimeMatrices(int** X,int filas,int columnas){
	int i,j;

	for ( i = 0; i < filas; i++){
			for( j=0;j<columnas;j++){
				printf("%d  ", X[i][j]);
			}
			printf("\n");
	}
}

void multiplicaMatrices(int** X,int filX,int colX,int** Y,int filY,int colY,int** Z)
{
	int i,j,k;
	int suma=0;

	for(i=0;i<filX;i++){//la multiplicación se realiza entre las filas de la primera matriz
		for(j=0;j<colY;j++){//y las columnas de la segunda matriz
			suma=0;//reiniciamos una vez puesto el valor correspondiente en la matriz Z
			for(k=0;k<filY;k++){
				suma=suma+X[i][k]*Y[k][j];
			}
			Z[i][j]=suma;//este es el valor de la multiplicación
		}
	}
}

int** init(int** X,int rows,int cols){

	int i;

	X=(int**)malloc(rows*sizeof(int*));//reservamos memoria

	for(i=0;i<rows;i++){
		X[i]=(int*)malloc(cols*sizeof(int*));
	}
	return X;

}

int main(){

	int filA,colA,filB,colB;

	FILE *matrizA;
	FILE *matrizB;

	matrizA = fopen("matrizA.txt","r");
	matrizB = fopen("matrizB.txt","r");

	fscanf (matrizA,"%d",&filA);
	fscanf (matrizA,"%d",&colA);

	fscanf (matrizB,"%d",&filB);
	fscanf (matrizB,"%d",&colB);

	if(colA!=filB){
		printf(" No se pueden multiplicar las matrices, el numero de columnas de la primer matriz debe ser igual al numero de filas de la segunda matriz!!! \n");
		return 1;
	}

	int** A;int** B;int** Result;


	A = init(A,filA,colA);
	B = init(B,filB,colB);
	Result = init(Result,filA,colB);

	llenaMatrices(matrizA,A,filA,colA);
	printf(" Matriz A: \n");
	imprimeMatrices(A,filA,colA);
	

	llenaMatrices(matrizB,B,filB,colB);
	printf(" Matriz B: \n");
	imprimeMatrices(B,filB,colB);

	multiplicaMatrices(A,filA,colA,B,filB,colB,Result);
	printf(" Multiplicación: \n");
	imprimeMatrices(Result,filA,colB);
	

	free(*A);free(*B);free(*Result);//liberamos memoria
	free(A);free(B);free(Result);

	return 0;



  }
