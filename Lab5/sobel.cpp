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

cxxpool::thread_pool pool{4};

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
    clock_t t;  

    // read through each frame and compute sobel value
    while(!frame.empty())
    {
        t = clock();
        to422_sobel_threaded(frame);
        waitKey(1);
        vid.read(frame);
        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("Frame took: %fs\n", time_taken);
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

    // // create threads and call sobel function
    // for(int i = 0; i < NUM_THREADS; i++)
    // {
    //     thread_arg[i].thread_num = i;
    //     //to442_grayscale_sobel((void *)&thread_arg[i]);
    //     pool.push(to442_grayscale_sobel, (void *)&thread_arg[i]);
    //     //pthread_create(&threads[i], NULL, to442_grayscale_sobel, (void *)&thread_arg[i]);
    // }

    auto future1 = pool.push(to442_grayscale_sobel, 0);
    auto future2 = pool.push(to442_grayscale_sobel, 1);
    auto future3 = pool.push(to442_grayscale_sobel, 2);
    auto future4 = pool.push(to442_grayscale_sobel, 3);

    future1.wait();
    quadrants[0].copyTo(output(Rect(0,            0,              image.cols/2, image.rows/2)));

    future2.wait();
    quadrants[1].copyTo(output(Rect(image.cols/2, 0,              image.cols/2, image.rows/2)));
    future3.wait();
    quadrants[2].copyTo(output(Rect(0,            image.rows/2,   image.cols/2, image.rows/2)));
    future4.wait();
    quadrants[3].copyTo(output(Rect(image.cols/2, image.rows/2,   image.cols/2, image.rows/2)));


    // wait for threads to finish
    // for(int i = 0; i < NUM_THREADS; i++)
    // {
    //     pthread_join(threads[i], NULL);
    // }

    // stitch frames back together
    
    
    imshow("Video", output);
 
}

void to442_grayscale_sobel(int index)
{
    Mat grayscale;

    // compute grayscale and sobel
    grayscale = to442_grayscale(quadrants[index]);
    quadrants[index] = to442_sobel(grayscale);

    //pthread_exit(NULL);
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

    float32_t blue[NUM_VECTORS];
    float32_t red[NUM_VECTORS];
    float32_t green[NUM_VECTORS];
    float32_t output[NUM_VECTORS];

    float32x4_t blues;
    float32x4_t reds;
    float32x4_t greens;
    float32x4_t outputs;

    // loop through every pixel starting at top left
    for(int i = 0; i < image.rows; i++)
    {
        // loop through each column in row in groups of 4
        // needs to be in groups of 4 to fit vector sizing
        for(int j = 0; j < image.cols; j+=4)
        {
            for(int k = 0; k < NUM_VECTORS && (j + k) < image.cols; k++)
            {
                // get the color of the pixel from original image
                Vec3b & color = image.at<Vec3b>(i,(j + k));

                // colors are in BRG format
                blue[k] = color[0];
                red[k] = color[0];
                green[k] = color[0];
            }

            blues = vld1q_f32(blue);
            reds = vld1q_f32(red);
            greens = vld1q_f32(green);

            blues = vmulq_n_f32(blues, BLUE_SCALAR);
            reds = vmulq_n_f32(reds, BLUE_SCALAR);
            greens = vmulq_n_f32(greens, BLUE_SCALAR);

            outputs = vaddq_f32(blues, reds);
            outputs = vaddq_f32(greens, outputs);

            vst1q_f32(output, outputs);

            for(int k = 0; k < NUM_VECTORS && (j + k) < image.cols; k++)
            {
                // add grayscale to new single channel image
                grayscale_image.at<uint8_t>(Point(j + k,i)) = (uint8_t)output[k];
            }
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
    const int8_t G_x[] = { -1, 0, 1, 
                        -2, 2,
                        -1, 0, 1 };

    const int8_t G_y[] = {  1,  2,  1, 
                         0, 0, 
                        -1, -2, -1};

    int x_conv = 0;
    int y_conv = 0;

    // get all the color values for each of the surrounding pixels
    uint8_t A = image.at<uint8_t>(i - 1,j -1);
    uint8_t B = image.at<uint8_t>(i,j - 1);
    uint8_t C = image.at<uint8_t>(i + 1,j - 1);
    uint8_t D = image.at<uint8_t>(i + 1,j);
    uint8_t E = image.at<uint8_t>(i + 1,j + 1);
    uint8_t F = image.at<uint8_t>(i,j + 1);
    uint8_t G = image.at<uint8_t>(i - 1,j + 1);
    uint8_t H = image.at<uint8_t>(i - 1,j);

    // create array of pixels
    uint8_t pixels[] = {A, B, C, D, E, F, G, H};

    // load pixels and other matricies into vector
    uint8x8_t vec_pixels = vld1_u8(pixels);
    int8x8_t vec_G_x = vld1_s8(G_x);
    int8x8_t vec_G_y = vld1_s8(G_y);

    // convert to int16 for multiplication 
    uint16x8_t mult_vec_pixels = vmovl_u8(vec_pixels);
    int16x8_t mult_vec_G_x = vmovl_s8(vec_G_x);
    int16x8_t mult_vec_G_y = vmovl_s8(vec_G_y);

    // convert unsigned to signed for multiplication
    int16x8_t s_mult_vec_pixels = vreinterpretq_s16_u16(mult_vec_pixels);

    // do the convolutions, cap at maximum value, should not be reached
    int16x8_t conv_1 = vmulq_s16(s_mult_vec_pixels, mult_vec_G_x);
    int16x8_t conv_2 = vmulq_s16(s_mult_vec_pixels, mult_vec_G_y);

    int16_t x_vals[NUM_PIXELS];
    int16_t y_vals[NUM_PIXELS]; 

    // store values back into array
    vst1q_s16(x_vals, conv_1);
    vst1q_s16(y_vals, conv_2);

    for(int i = 0; i < NUM_PIXELS; i++)
    {
        x_conv += x_vals[i];
        y_conv += y_vals[i];
    } 

    // apply absolute value and combine
    uint sobel_value = abs(x_conv) + abs(y_conv);

    // thresholding for uint8_t
    if(sobel_value > 255)
        sobel_value = 255;

    return (uint8_t)sobel_value;

}