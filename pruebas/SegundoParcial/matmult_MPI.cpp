#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

using namespace std;

void multMatCUDA(double *M_a, double *M_b, double *R_c, int NRA, int NCA,
                 int NCB);

#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

void multMatMPI(double *a, double *b, double *c, int NRA, int NCA, int NCB) {
  for (int k = 0; k < NCB; ++k) {
    for (int i = 0; i < NRA; ++i) {
      for (int j = 0; j < NCA; ++j) {
        c[i * NCB + k] += a[i * NCA + j] * b[j * NCB + k];
      }
    }
  }
}

bool compareTo(double *c, double *d_c, int H, int W) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      if (c[i * W + j] != d_c[i * W + j]) {
        return false;
      }
    }
  }
  return true;
}

bool compare(double *d_c, int H, int W, int NCA) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      if (NCA != d_c[i * W + j]) {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  int numtasks,   /* number of tasks in partition */
      taskid,     /* a task identifier */
      numworkers, /* number of worker tasks */
      source,     /* task id of message source */
      dest,       /* task id of message destination */
      mtype,      /* message type */
      elements,   /* elements of matrix A sent to each worker */
      averow, extra,
      offset,      /* used to determine elements sent to each worker */
      i, j, k, rc; /* misc */

  clock_t start, end;
  double time_used;

  int NRA = 12000;
  int NCA = 12000;
  int NCB = 12000;

  double *a, /* matrix A to be multiplied */
      *b,    /* matrix B to be multiplied */
      *c,    /* result matrix  C in MPI*/
      *d_c;  /* result matrix C in CUDA*/

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (numtasks < 2) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  numworkers = numtasks - 1;

  /**************************** master task
   * **************************************/
  if (taskid == MASTER) {
    a = (double *)malloc(NRA * NCA * sizeof(double));
    b = (double *)malloc(NCA * NCB * sizeof(double));
    c = (double *)malloc(NRA * NCB * sizeof(double));
    d_c = (double *)malloc(NRA * NCB * sizeof(double));

    printf("mpi_mm has started with %d tasks.\n", numtasks);
    printf("Initializing arrays...\n");
    double cont = 1;
    for (i = 0; i < NRA; i++) {
      for (j = 0; j < NCA; j++) {
        a[i * NCA + j] = 1;
      }
    }

    cont = 1;
    for (i = 0; i < NCA; i++) {
      for (j = 0; j < NCB; j++) {
        b[i * NCB + j] = 1;
      }
    }

    start = clock();

    /* Send matrix data to the worker tasks */
    averow = NRA / numworkers;
    extra = NRA % numworkers;
    offset = 0;
    mtype = FROM_MASTER;
    for (dest = 1; dest <= numworkers; dest++) {
      if (dest <= extra) {
        elements = averow + 1;
      } else {
        elements = averow;
      }
      printf("Sending %d elements to task %d offset=%d\n", elements, dest,
             offset);
      // Fila de inicio
      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      // Numero de Filas
      MPI_Send(&elements, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      // Filas Matriz A
      MPI_Send(&a[offset * NCA], elements * NCA, MPI_DOUBLE, dest, mtype,
               MPI_COMM_WORLD);
      // Matriz B
      MPI_Send(b, NCA * NCB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
      offset = offset + elements;
    }

    /* Receive results from worker tasks */
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&elements, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset * NCB], elements * NCB, MPI_DOUBLE, source, mtype,
               MPI_COMM_WORLD, &status);
      MPI_Recv(&d_c[offset * NCB], elements * NCB, MPI_DOUBLE, source, mtype,
               MPI_COMM_WORLD, &status);
      printf("Received results from task %d\n", source);
    }

    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo invertido = %lf s\n", time_used);

    /* Print results */
    // printf("******************************************************\n");
    // printf("Result Matrix with MPI:\n");
    //
    // for (i = 0; i < NRA; i++) {
    //   for (j = 0; j < NCB; j++) {
    //     cout << c[i * NCB + j] << " ";
    //   }
    //   cout << endl;
    // }
    // printf("Result Matrix with CUDA:\n");
    // for (i = 0; i < NRA; i++) {
    //   for (j = 0; j < NCB; j++) {
    //     cout << d_c[i * NCB + j] << " ";
    //   }
    //   cout << endl;
    // }
    //
    // printf("\n******************************************************\n");
    // printf("Done.\n");
    // if (compareTo(c, d_c, elements, NCB))
    //   cout << "Funciona!" << endl;
    // else {
    //   cout << "No Funciona" << endl;
    // }
    if (compare(c, NRA, NCB, NCA))
      cout << "Funciona!" << endl;
    else {
      cout << "No Funciona" << endl;
    }
  }

  /**************************** worker task
   * *************************************/
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

    a = (double *)malloc(elements * NCA * sizeof(double));
    b = (double *)malloc(NCA * NCB * sizeof(double));
    c = (double *)malloc(elements * NCB * sizeof(double));
    d_c = (double *)malloc(elements * NCB * sizeof(double));

    MPI_Recv(a, elements * NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD,
             &status);
    MPI_Recv(b, NCA * NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

    multMatMPI(a, b, c, elements, NCA, NCB);

    // multMatCUDA(a, b, d_c, elements, NCA, NCB);

    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(c, elements * NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(d_c, elements * NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  free(a);
  free(b);
  free(c);
  free(d_c);
}
