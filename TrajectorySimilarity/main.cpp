#include<iostream>
#include<opencv/cv.hpp>
#include<opencv/highgui.h>
#include "StipDetector.h"

using namespace std;
using namespace cv;




int main (int argc, char** argv)
{

    // argument transfering
    if(argc != 5)
    {
        std::cout << "./program [filename] [sigma] [finest scale] [n_level]"<< std::endl;
        return -1;
    }
    char* filename = argv[1];
    double sigma = atof(argv[2]);
    double rho = atof(argv[3]);
    int n_level = atoi(argv[4]);


    // video IO
    cv::VideoCapture cap(filename);
    if(!cap.isOpened())
    {
        std::cout<< "video is not opened!"<< std::endl;
        return -2;
    }

    // variable configuration
    std::vector<cv::KeyPoint> corners;
    cv::Mat frame,frame_current, frame_previous;





    // main loop
    int t=0;
    while(1)
    {
        // release the corner vector
        corners.clear();

        // read current frame from the video
        cap >> frame;

        // convert the type of the frame from uchar to double
        cv::cvtColor(frame, frame_current, cv::COLOR_BGR2GRAY);
        frame_current.convertTo(frame_current,CV_64F);


        // if it is the end of the frame, break the loop
        if (frame_current.cols==0 || frame_current.rows==0)
        {
            cout << "video ends." <<endl;
            break;
        }


        // mirroring boundary condition and Gaussian smooth
        GaussianBlur(frame_current, frame_current, cv::Size(0,0), sigma,sigma, BORDER_REFLECT);


        if(t>0)
        {

            /// scale-space representation, rho_s = pow(1.5,s)*rho, s = 1,2,3,...,5
            for (s = 0; s < 5; s++)
            {
                MotionTensorScore (frame_current, frame_previous, score, rho*std::pow(1.6,s));

                // find local maximum
                Dilation(score, score_dilate,11);
                score_peak = score_dilate-score;
                score_peak.convertTo(score_peak, CV_32F);
                //cv::threshold(score_peak, score_peak, 0,255.0,cv::THRESH_BINARY );

                // find ROI
                score_roi = abs(score);
                minMaxLoc(score_roi,&min_val,&max_val);
                score_roi -= min_val;
                convertScaleAbs(score_roi,score_roi,255/(max_val-min_val));
                cv::threshold(score_roi, score_roi,10,255,cv::THRESH_BINARY_INV);
                score_roi.convertTo(score_roi,CV_32F);

                // extract key points
                for (j = 0; j < score.rows; j++)
                {

                    float* ptr_score_roi = (float*) score_roi.ptr(j);
                    float* ptr_score_peak = (float*) score_peak.ptr(j);
                    for (i = 0; i < score.cols; i++)
                    {
                        if(ptr_score_roi[i] + ptr_score_peak[i] == 0.0f)
                            corners.push_back(KeyPoint(Point2f(i,j),std::pow(2,s)*rho) );

                    }
                }


            }

            cv::Mat frame_middle = (frame_current + frame_previous)/2.0;
            VideoKeypointDisplay(frame_middle, corners);

        }



        // for the next frame
        frame_current.copyTo(frame_previous);
        t++;

    }




    return 0;

}
