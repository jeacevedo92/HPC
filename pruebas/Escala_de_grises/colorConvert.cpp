#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
#define CHANNELS 3

__global__ void colorConvert(unsigned char *grayImage, unsigned char *rgbImage,
                             int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < width && y < height) {
    // get 1D coordinate for the grayscale image
    int grayOffset = y * width + x;
    // one can think of the RGB image having
    // CHANNEL times columns than the gray scale image
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char b = rgbImage[rgbOffset];     // red value for pixel
    unsigned char g = rgbImage[rgbOffset + 2]; // green value for pixel
    unsigned char r = rgbImage[rgbOffset + 3]; // blue value for pixel
    // perform the rescaling and store it
    // We multiply by floating point constants
    grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

void colorConvertSec(unsigned char *image, unsigned char *image_Gray, int width,
                     int height) {

  int grayOffset = 0, rgbOffset = 0;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      grayOffset = i * width + j;
      rgbOffset = grayOffset * CHANNELS;
      unsigned char b = image[rgbOffset];     // red value for pixel
      unsigned char g = image[rgbOffset + 2]; // green value for pixel
      unsigned char r = image[rgbOffset + 3]; // blue value for pixel
      image_Gray[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
  }
}

int main(int argc, char **argv) {
  Mat image;
  image = imread("./inputs/img2.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  Size s = image.size();
  unsigned char *image_Gray = new unsigned char[s.height * s.width];
  int width = s.width;
  int height = s.height;

  unsigned char *d_image, *d_image_Gray, *h_imageOutput;
  int size = sizeof(unsigned char) * width * height * image.channels();
  int sizeGray = sizeof(unsigned char) * width * height;
  h_imageOutput = (unsigned char *)malloc(sizeGray);
  cudaMalloc((void **)&d_image, size);
  cudaMalloc((void **)&d_image_Gray, sizeGray);

  colorConvertSec(image.data, image_Gray, width, height);

  Mat grayImg;
  grayImg.create(s.height, s.width, CV_8UC1);
  grayImg.data = image_Gray;

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)),
               1);

  // cudaMemcpy(d_image_Gray, image_Gray, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_image, image.data, size, cudaMemcpyHostToDevice);

  colorConvert<<<dimGrid, dimBlock>>>(d_image_Gray, d_image, width, height);
  cudaMemcpy(h_imageOutput, d_image_Gray, sizeGray, cudaMemcpyDeviceToHost);

  Mat grayImgCuda;
  grayImgCuda.create(s.height, s.width, CV_8UC1);
  grayImgCuda.data = h_imageOutput;

  if (!image.data) // Check for invalid input
  {
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  imwrite("./outputs/1088331150.png", grayImgCuda);
  return 0;
}
