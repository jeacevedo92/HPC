/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 04/13/05
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define NEA 1000000                 /* number of elements in matrix A, B and C */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	elements,                  /* elements of matrix A sent to each worker */
	averow, extra, offset, /* used to determine elements sent to each worker */
	i, j, k, rc;           /* misc */

double *a,           /* matrix A to be multiplied */
	*b,           /* matrix B to be multiplied */
	*c;           /* result matrix C */

MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;

   a = (double*)malloc(NEA*sizeof(double));
   b = (double*)malloc(NEA*sizeof(double));
   c = (double*)malloc(NEA*sizeof(double));


/**************************** master task ************************************/
   if (taskid == MASTER)
   {
      printf("mpi_mm has started with %d tasks.\n",numtasks);
      printf("Initializing arrays...\n");
      for (i=0; i<NEA; i++) {
         a[i] = 1;
         b[i] = 1;
      }

      /* Send matrix data to the worker tasks */
      averow = NEA / numworkers;
      extra = NEA % numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest = 1; dest <= numworkers; dest++)
      {
         if(dest <= extra){
            elements = averow + 1;
         }
         else{
            elements = averow;
         }   	
         printf("Sending %d elements to task %d offset=%d\n",elements,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&elements, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset], elements, MPI_DOUBLE, dest, mtype,
                   MPI_COMM_WORLD);
         MPI_Send(&b[offset], elements, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + elements;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&elements, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset], elements, MPI_DOUBLE, source, mtype, 
                  MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
      }

      /* Print results */
      printf("******************************************************\n");
      printf("Result Matrix:\n");
      // for (i=0; i<NEA; i++)
      // {
      //    printf("%.2f ", c[i]);
      // }
      printf("\n******************************************************\n");
      printf ("Done.\n");
   }


/**************************** worker task ************************************/
   if (taskid > MASTER)
   {
      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

      MPI_Recv(&a[offset], elements, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&b[offset], elements, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

      for (k=offset; k<NEA; k++) {
         c[k] = a[k] + b[k];
      }
      
      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&c[offset], elements, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }
   MPI_Finalize();
   free(a);
   free(b);
   free(c);
}
