#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;     // allows use of opencv functions without cv::

// (0.2126R + 0.7152G + 0.0722B)
#define BLUE_SCALAR     0.0722
#define RED_SCALAR      0.2126
#define GREEN_SCALAR    0.7152

Mat to442_sobel(Mat image);
Mat to442_grayscale(Mat image);
uint8_t calculate_sobel_val(Mat image, int i, int j);
