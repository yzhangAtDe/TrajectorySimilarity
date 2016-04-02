#include<iostream>
#include<opencv/cv.hpp>
#include<opencv/highgui.h>
#include "StipDetector.h"

using namespace std;
using namespace cv;


std::vector<cv::KeyPoint> corners;
cv::Mat frame,frame_current, frame_previous;
cv::Mat roi;


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

    //detector configuration
    StipDetector detector;
    detector.SetFineScale(rho);
    detector.SetLevelNum( n_level);
    detector.SetMethodScore(StipDetector::MinEigen);
    detector.SetMethodROI(StipDetector::GM2);




    // main loop
    int t=0;
    while(1)
    {
        // release the corner vector
        corners.clear();

        // read current frame from the video
        cap >> frame;

        // convert the type of the frame from uchar to double
        cv::resize(frame,frame,Size(200,150));
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
            // detector configuration
            detector.SetFrames(frame_previous, frame_current);
            detector.detect();
            detector.GetROI(roi);
//            detector.GetKeyPoints(corners);
            detector.VideoKeypointDisplay();
            detector.ClearPoints();
            imshow("roi", roi);
            waitKey(10);

        }



        // for the next frame
        frame_current.copyTo(frame_previous);
        t++;

    }




    return 0;

}
