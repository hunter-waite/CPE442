#include <iostream>
#include <cmath>
#include <pthread.h>
#include <arm_neon.h>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cxxpool.hpp"

using namespace std;
using namespace cv;     // allows use of opencv functions without cv::

// (0.2126R + 0.7152G + 0.0722B)
#define BLUE_SCALAR     .0722
#define RED_SCALAR      .2126
#define GREEN_SCALAR    .7152

#define EDGE_BUFFER     2
#define NUM_THREADS     4
#define NUM_PIXELS      8
#define NUM_VECTORS     4
// global variable for each quadrant
Mat quadrants[NUM_THREADS];

// tells the thread what to pass into function
struct thread_args {
    int thread_num;
};

void to422_sobel_threaded(Mat image);
void to442_grayscale_sobel(int image);
Mat to442_sobel(Mat image);
Mat to442_grayscale(Mat image);
uint8_t calculate_sobel_val(Mat image, int i, int j);