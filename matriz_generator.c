#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

void setNumbers(FILE *stream, int rows, int cols){
	int i, j;
	for(i=0; i<rows; i++){
		for(j=0; j<cols; j++){
			if(j == cols-1)
				fprintf(stream, "%d", rand()%50);
			else
				fprintf(stream, "%d,", rand()%50);
		}
		if(i != rows-1)
			fprintf(stream, "%s\n", "");
	}
}

//Generate a matriz of rowsxcols with integers numbers between 0 and max

int main(int argc, char** argv)
{

	int rows1, cols1, rows2, cols2, i, j;;

	printf("Rows and Cols of matriz 1\n");
	scanf("%d %d", &rows1, &cols1);
	printf("Rows and Cols of matriz 2\n");
	scanf("%d %d", &rows2, &cols2);

	srand(time(NULL));	
	FILE *stream, *stream2;
	stream = fopen("m1.txt", "w");
	fprintf(stream, "%d\n", rows1);
	fprintf(stream, "%d\n", cols1);
	setNumbers(stream, rows1, cols1);

	stream2 = fopen("m2.txt", "w");
	fprintf(stream2, "%d\n", rows2);
	fprintf(stream2, "%d\n", cols2);
	setNumbers(stream2, rows2, cols2);

	fclose(stream);
	fclose(stream2);

	return 0;
}