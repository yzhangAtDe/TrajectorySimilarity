#ifndef STIPDETECTOR_H
#define STIPDETECTOR_H

#include<iostream>
#include<vector>
#include<highgui.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class StipDetector
{
    public:
        enum RoiMethod {ScoreThreshold, GM2};
        enum ScoreMethod {Harris, MinEigen};

        StipDetector();
        StipDetector(const Mat& frame_current, const Mat& frame_previous);
        virtual ~StipDetector();
        void VideoKeypointDisplay( Mat& frame, const vector<KeyPoint>& corners);

        void SetFineScale(double scale);
        void SetLevelNum( int n_level);
        void SetMethodScore(StipDetector::ScoreMethod method);
        void SetMethodROI(StipDetector::RoiMethod method);
        void SetFrames(const cv::Mat& f1, const cv::Mat& f2);

        void detect();
        void ClearPoints();

        void GetROI(cv::Mat& fg);
        void GetScore(cv::Mat& score);
        void GetKeyPoints(vector<cv::KeyPoint>& corners);
        void VideoKeypointDisplay( );

    protected:
        void DefineROI();
        void Gradient (const Mat& src, Mat& gradx, Mat& grady, bool use_sobel = true);
        void MotionTensorScore (const cv::Mat& frame_current, const cv::Mat& frame_previous, cv::Mat& score, double rho);
        void Dilation(const cv::Mat& src, cv::Mat& dst, int kernelsize=5);
        void Open(const cv::Mat& src, cv::Mat& dst, int kernelsize=5);
        void Close(const cv::Mat& src, cv::Mat& dst, int kernelsize=5);



    private:
        int _n_level; // number of scale-space levels (or Pyramid levels)
        double _scale_base; // the finest scale, i.e. size of gaussian kernel
        std::vector<cv::KeyPoint> _corners; // keypoints
        Mat _frame_current; // f2
        Mat _frame_previous; // f1
        RoiMethod _roi_method;
        ScoreMethod _score_method;

        Mat _score, _score_dilate, _score_peak;
        Mat _roi;
        cv::Ptr<cv::BackgroundSubtractorMOG2> _pMOG;


};

#endif // STIPDETECTOR_H
