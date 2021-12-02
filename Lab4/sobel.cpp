/**********************************************************************
 * File: sobel.cpp
 * 
 * Description: Take in an image path from the command line and display
 *  it in a new window using opencv2
 * 
 * Authors: Hunter Waite, Jeremy Szeto, Neil Patel
 * 
 * Revision History: 
 * 
 * Usage: sobel [path_to_video]
 * ********************************************************************/
#include "sobel.hpp"

int main(int argc, char *argv[]) {

    VideoCapture vid;
    Mat frame, sobel_frame;

    // check arguments
    if(argc != 2) {
        cout << "Incorrect number of arguments\n" <<
        "Usage: loadImage [path_to_image]\n";
        exit(1);
    }

    vid.open(argv[1]);
    if(vid.isOpened()==0)
    {
        cout<<"The video file cannot be opened."<<endl;
        return -1;
    }

    vid.read(frame);

    // read through each frame and compute sobel value
    while(!frame.empty())
    {
        to422_sobel_threaded(frame);
        waitKey(1);
        vid.read(frame);
    }

}

void to422_sobel_threaded(Mat image)
{
    Mat grayscale_frame, sobel_frame, workhorse;

    Mat output(image.rows, image.cols, CV_8UC1, Scalar(0));

    pthread_t threads[NUM_THREADS];
    struct thread_args thread_arg[4];

    // split up into quadrants
    workhorse = image.clone();

    // make sure each quadrant has enough edge for sobel to fill out
    quadrants[0] = workhorse(Rect(0, 0, image.cols/2 + EDGE_BUFFER, image.rows/2 + EDGE_BUFFER));
    quadrants[1] = workhorse(Rect(image.cols/2 - EDGE_BUFFER, 0, image.cols/2 + EDGE_BUFFER, image.rows/2 + EDGE_BUFFER));
    quadrants[2] = workhorse(Rect(0, image.rows/2 - EDGE_BUFFER, image.cols/2 + EDGE_BUFFER, image.rows/2 + EDGE_BUFFER));
    quadrants[3] = workhorse(Rect(image.cols/2 - EDGE_BUFFER, image.rows/2 - EDGE_BUFFER, image.cols/2 + EDGE_BUFFER, image.rows/2 + EDGE_BUFFER));

    // create threads and call sobel function
    for(int i = 0; i < NUM_THREADS; i++)
    {
        thread_arg[i].thread_num = i;
        pthread_create(&threads[i], NULL, to442_grayscale_sobel, (void *)&thread_arg[i]);
    }

    // wait for threads to finish
    for(int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // stitch frames back together
    quadrants[0].copyTo(output(Rect(0,            0,              image.cols/2, image.rows/2)));
    quadrants[1].copyTo(output(Rect(image.cols/2, 0,              image.cols/2, image.rows/2)));
    quadrants[2].copyTo(output(Rect(0,            image.rows/2,   image.cols/2, image.rows/2)));
    quadrants[3].copyTo(output(Rect(image.cols/2, image.rows/2,   image.cols/2, image.rows/2)));

    imshow("Video", output);
 
}

void *to442_grayscale_sobel(void *index)
{
    Mat grayscale;

    // get correct quadrant index
    int i = (*(struct thread_args *)index).thread_num;

    // compute grayscale and sobel
    grayscale = to442_grayscale(quadrants[i]);
    quadrants[i] = to442_sobel(grayscale);

    pthread_exit(NULL);
}

Mat to442_sobel(Mat image)
{
    Mat sobel_image(image.rows - 2, image.cols - 2, CV_8UC1, Scalar(0));

    // check matrix size
    if(image.rows < 3 || image.cols < 3)
    {
        cout << "Cannot perform sobel on less than 3x3 matrix";
        exit(1);
    }

    // loop through all pixels in mat and compute sobel value
    for(int i = 1; i < image.rows - 1; i++)
    {
        for(int j = 1; j < image.cols - 1; j++)
        {
            uint8_t color = calculate_sobel_val(image, i, j);

            sobel_image.at<uint8_t>(Point(j - 1,i - 1)) = color;
        }
    }

    return(sobel_image);
}


/* Simple Conversion function from opencv BRG to single channel (grayscale) */
Mat to442_grayscale(Mat image)
{
    // create new single channel image (grayscale) with all values set to black
    // it is the same size as the original image
    Mat grayscale_image(image.rows, image.cols, CV_8UC1, Scalar(0));

    // loop through every pixel starting at top left
    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            // get the color of the pixel from original image
            Vec3b & color = image.at<Vec3b>(i,j);

            // color vector is in BRG format
            // (0.2126R + 0.7152G + 0.0722B)
            uint8_t grayscale_val = (color[0] * BLUE_SCALAR) + (color[1] * RED_SCALAR) + (color[2] * GREEN_SCALAR);

            // add grayscale to new single channel image
            grayscale_image.at<uint8_t>(Point(j,i)) = grayscale_val;
        }
    }

    return(grayscale_image);
}


/* A simple implementation of the sobel filter on a single pixel
 * Schema for matrix definitions
 *  A B C
 *  H _ D
 *  G F E
 */
uint8_t calculate_sobel_val(Mat image, int i, int j) 
{
    // scalars used for filters
    const int G_x[] = { -1, 0, 1, 
                        -2, 0, 2,
                        -1, 0, 1 };

    const int G_y[] = {  1,  2,  1, 
                         0,  0,  0, 
                        -1, -2, -1};

    // get all the color values for each of the surrounding pixels
    uint8_t A = image.at<uint8_t>(i - 1,j -1);
    uint8_t B = image.at<uint8_t>(i,j - 1);
    uint8_t C = image.at<uint8_t>(i + 1,j - 1);
    uint8_t D = image.at<uint8_t>(i + 1,j);
    uint8_t E = image.at<uint8_t>(i + 1,j + 1);
    uint8_t F = image.at<uint8_t>(i,j + 1);
    uint8_t G = image.at<uint8_t>(i - 1,j + 1);
    uint8_t H = image.at<uint8_t>(i - 1,j);

    // calculate the convolution for each of the matricies
    int x_conv = A * G_x[0] + C * G_x[2] + H * G_x[3] + D * G_x[5] + G * G_x[6] + E * G_x[8];
    int y_conv = A * G_y[0] + B * G_y[1] + C * G_y[2] + G * G_y[6] + F * G_y[7] + E * G_y[8];

    // apply absolute value and combine
    uint sobel_value = abs(x_conv) + abs(y_conv);

    // thresholding for uint8_t
    if(sobel_value > 255)
        sobel_value = 255;

    return (uint8_t)sobel_value;

}