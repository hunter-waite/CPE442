/**********************************************************************
 * File: loadImage.cpp
 * 
 * Description: Take in an image path from the command line and display
 *  it in a new window using opencv2
 * 
 * Authors: Hunter Waite, Jeremy Szeto, Neil Patel
 * 
 * Revision History: 
 * 
 * Usage: loadImage [path_to_image]
 * ********************************************************************/
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;     // allows use of opencv functions without cv::

int main(int argc, char *argv[]) {
    // check arguments
    if(argc != 2) {
        cout << "Incorrect number of arguments\n" <<
        "Usage: loadImage [path_to_image]\n";
        exit(1);
    }

    // try and open image
    Mat image = imread(argv[1], IMREAD_COLOR); // loads an image into a mat

    // check and make sure the file was opened
    if(image.empty()) {
        cout << "Could not find/open image " << argv[1] << "\n" <<
        "Usage: loadImage [path_to_image]\n";
        exit(1);
    }

    // pull up the image in a window called Window Name
    imshow("Window Name", image);

    // wait for key press, will close after key is pressed
    waitKey(0);
    return 0;
}