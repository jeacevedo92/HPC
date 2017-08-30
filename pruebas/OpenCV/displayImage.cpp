#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
    return -1;
  }

  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file
  int rows = image.rows;
  int cols = image.cols;
  cout << "rows: " << rows << " cols: " << cols << endl;
  for (int i = 0; i < rows; i++) {
    cout << "[ ";
    for (int j = 0; j < cols; j++) {
      cout << j << " ";
    }
    cout << " ]";
    cout << endl;
  }

  if (!image.data) // Check for invalid input
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  namedWindow("Display window",
              WINDOW_AUTOSIZE);    // Create a window for display.
  imshow("Display window", image); // Show our image inside it.

  waitKey(0); // Wait for a keystroke in the window
  return 0;
}
