#ifndef STIPDETECTOR_H
#define STIPDETECTOR_H

#include<iostream>
#include<opencv/cv.hpp>
#include<opencv/highgui.h>

using namespace std;
using namespace cv;

class StipDetector
{
    public:
        StipDetector();
        StipDetector(const Mat& frame_current, const Mat& frame_previous);
        virtual ~StipDetector();
        void VideoKeypointDisplay( Mat& frame, const vector<KeyPoint>& corners);
        void Gradient (const Mat& src, Mat& gradx, Mat& grady, bool use_sobel = true);
        void MotionTensorScore (const cv::Mat& frame_current, const cv::Mat& frame_previous, cv::Mat& score, double rho);
        void SetFineScale(double scale);
        void SetLevelNum( int n_level);
        void detect();


    protected:
    private:
        int _n_level; // number of scale-space levels (or Pyramid levels)
        double _scale_base; // the finest scale, i.e. size of gaussian kernel
        vector<Keypont> _corners; // keypoints
        Mat _frame_current; // f2
        Mat _frame_previous; // f1


};

#endif // STIPDETECTOR_H
