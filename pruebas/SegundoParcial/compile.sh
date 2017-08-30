/usr/local/cuda/bin/nvcc matmult.cu -c matmult.o
mpic++ -c matmult_MPI.cpp -o matmult_MPI.o
mpic++ matmult.o matmult_MPI.o -o matmult -L/usr/local/cuda/lib64/ -lcudart
