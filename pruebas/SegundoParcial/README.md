# INFORME SEGUNDO PARCIAL

Para desarrollar este parcial se procedio a divir el problema en dos fases, la primera la lógica de la multiplicacion 
de matrices en CUDA, y la segunda la lógica de la multiplicación de matrices con MPI.

Para el programa realizado en CUDA, se reciben los siguientes elementos:

``` cpp
    void multMatCUDA(double *M_a, double *M_b, double *R_c, int NRA, int NCA, int NCB){
                 ...
    } 
```
En donde: 

1. **M_a:** Es la sub-matriz A.
2. **M_b:** Es la matriz B.
3. **R_c:** Es la matriz C en donde se guardaran los resultados de la multiplicación.
4. **NRA:** Numero de filas de la sub-matriz A.
5. **NCA:** Numero de columnas de la sub-matriz A.
6. **NCB:** Numero de columnas de la matriz B.

Dentro de este se pasan los datos al device, y posteriomente se procede a llamar a la funcion del kernel, la cual ejecuta la multiplicacion de matrices en paralelo con CUDA.

Para el programa realizado en MPI se procede a asignar tareas a los diferentes nodos workers y posteriormente dentro de estos se llama a la función de multiplicacion con MPI o con CUDA, 
cuando los resultados han sido generados se procede a enviar de nuevo al master para poder visualizar los resultados.

La funcion de multiplicacion con MPI recibe los siguientes elementos:

``` cpp
void multMatMPI(double *a, double *b, double *c, int NRA, int NCA, int NCB) {
...
}
```

1. **a:** Es la sub-matriz A.
2. **b:** Es la matriz B.
3. **c:** Es la matriz C en donde se guardaran los resultados de la multiplicación.
4. **NRA:** Numero de filas de la sub-matriz A.
5. **NCA:** Numero de columnas de la sub-matriz A.
6. **NCB:** Numero de columnas de la matriz B.

Corremos los dos codigos en CUDA y en MPI, procedemos a generar los .o :
``` 
    /usr/local/cuda/bin/nvcc matmult.cu -c matmult.o
    mpic++ -c matmult_MPI.cpp -o matmult_MPI.o
```
Despues unimos ambos archivos .o con el siguiente comando:
```
    mpic++ matmult.o matmult_MPI.o -o matmult -L/usr/local/cuda/lib64/ -lcudart
```
Despues procedemos a crear el sbatch de la siguiente manera:
```
#!/bin/bash

#SBATCH --job-name=matmult_MPI
#SBATCH --output=matmult_MPI.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun matmult
```

Para ejecutar el sbatch escribimos el siguiente comando:

```
sbatch matmult_MPI.sh
```
para visualizar el resultado escribimos el siguiente comando:

```
cat matmult_MPI.out
```

# Analisis de Datos
