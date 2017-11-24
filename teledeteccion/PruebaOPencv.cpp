#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;

int main(int argc, char** argv){

char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 Mat gray_image;

 cvtColor( image, gray_image, CV_BGR2GRAY );

 imwrite( "./Gray_Image.jpg", gray_image );

 return 0;
}



