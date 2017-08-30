#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

__global__ void convolution_2D_basic_kernel(int *in, int *mask, int *out,
                                            int maskwidth, int w, int h) {
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  if (Col < w && Row < h) {
    int pixVal = 0;
    int N_start_col = Col - (maskwidth / 2);
    int N_start_row = Row - (maskwidth / 2);

    for (int j = 0; j < maskwidth; j++) {
      for (int k = 0; k < maskwidth; k++) {
        int curRow = N_start_row + j;
        int curCol = N_start_col + k;

        if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
          pixVal += in[curRow * w + curCol] * mask[j * maskwidth + k];
        }
      }
    }
    out[Row * w + Col] = pixVal;
  }
}

void convolution_2D_basic(int *in, int *mask, int *out, int maskwidth, int w,
                          int h) {
  for (int Col = 0; Col < w; Col++) {
    for (int Row = 0; Row < h; Row++) {
      int pixVal = 0;
      int N_start_col = Col - (maskwidth / 2);
      int N_start_row = Row - (maskwidth / 2);

      for (int j = 0; j < maskwidth; j++) {
        for (int k = 0; k < maskwidth; k++) {
          int curRow = N_start_row + j;
          int curCol = N_start_col + k;

          if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
            pixVal += in[curRow * w + curCol] * mask[j * maskwidth + k];
          }
        }
      }
      out[Row * w + Col] = pixVal;
    }
  }
}

void printMatrix(int *result, int h, int w) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      cout << result[i * w + j] << " ";
    }
    cout << endl;
  }
}

bool compareTo(int *h_c, int *h_result, int h, int w) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (h_c[i * w + j] != h_result[i * w + j]) {
        return false;
      }
    }
  }
  return true;
}

int main() {
  clock_t start, end;
  double cpu_time_used, gpu_time_used;
  float blockSize = 4;
  int *in, *mask, *out, *d_in, *d_mask, *d_out, *out_result;
  int h = 7, w = 7, maskwidth = 3;
  // Asignar memoria en el host
  in = (int *)malloc(sizeof(int) * h * w);
  mask = (int *)malloc(sizeof(int) * maskwidth * maskwidth);
  out = (int *)malloc(sizeof(int) * h * w);
  out_result = (int *)malloc(sizeof(int) * h * w);

  // Inicializar las matrices
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      in[i * w + j] = 1;
      out[i * w + j] = 1;
      out_result[i * w + j] = 1;
    }
  }
  for (int i = 0; i < maskwidth; i++) {
    for (int j = 0; j < maskwidth; j++) {
      mask[i * w + j] = i + 1;
    }
  }

  start = clock();
  // Llamar funcion que sume dos vectores y retorne el resultado en out
  convolution_2D_basic(in, mask, out, maskwidth, w, h);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido CPU = %lf s\n", cpu_time_used);

  // Asignacion de memoria en el device
  cudaMalloc(&d_in, sizeof(int) * h * w);
  cudaMalloc(&d_mask, sizeof(int) * maskwidth * maskwidth);
  cudaMalloc(&d_out, sizeof(int) * h * w);

  // Copiar los datos del host al device
  cudaMemcpy(d_in, in, h * w * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, mask, maskwidth * maskwidth * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(w / blockSize), ceil(h / blockSize), 1);

  start = clock();
  convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(d_in, d_mask, d_out,
                                                     maskwidth, w, h);
  cudaMemcpy(out_result, d_out, h * w * sizeof(int), cudaMemcpyDeviceToHost);
  end = clock();
  gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU = %lf s\n", gpu_time_used);

  printMatrix(out, h, w);
  cout << "matrix" << endl;
  printMatrix(out_result, h, w);
  if (compareTo(out, out_result, h, w)) {
    printf("Matrices Iguales");
  } else {
    printf("Matrices Diferentes");
  }
  cudaFree(d_in);
  cudaFree(d_mask);
  cudaFree(d_out);
  free(in);
  free(mask);
  free(out);
  free(out_result);
  return 0;
}
