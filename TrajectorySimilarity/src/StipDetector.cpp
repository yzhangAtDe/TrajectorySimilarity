#include <cv.h>
#include <highgui.h>
#include "StipDetector.h"
#define PI 3.1415926

using namespace cv;
using namespace std;



void StipDetector::Dilation(const cv::Mat& src, cv::Mat& dst, int kernelsize)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation
    cv::dilate( src, dst, element );

}

void StipDetector::Open(const cv::Mat& src, cv::Mat& dst, int kernelsize)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation
    cv::erode( src, dst, element );
    cv::dilate(dst,dst,element);

}


void StipDetector::Close(const cv::Mat& src, cv::Mat& dst, int kernelsize)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation
    cv::dilate( src, dst, element );
    cv::erode(dst,dst,element);

}


void StipDetector::Gradient (const Mat& src, Mat& gradx, Mat& grady, bool use_sobel )
{

    if (use_sobel)
    {
        Sobel(src,gradx, -1, 1,0,1, 1,0, BORDER_REFLECT);
        Sobel(src,grady, -1, 0,1,1, 1,0, BORDER_REFLECT);
    }
    else
    {
        int i,j;
        const int ny = src.rows;
        const int num_channel = src.channels();
        const int nx = src.cols * num_channel;

        gradx = src.clone();
        grady = src.clone();


        for(j = 1; j < ny-1; j++)
        {
            const double* ptr      = src.ptr<double>(j);
            const double* ptr_up   = src.ptr<double>(j+1);
            const double* ptr_down = src.ptr<double>(j-1);

            double* ptr_gradx = (double*) gradx.ptr(j);
            double* ptr_grady = (double*) grady.ptr(j);

            for(i = num_channel; i < nx-num_channel; i++)
            {
                ptr_gradx[i] = (ptr[i+num_channel] - ptr[i-num_channel])/2.0;
                ptr_grady[i] = (ptr_down[i] - ptr_up[i])/2.0;

            }
        }
    }
}




void StipDetector::MotionTensorScore (const cv::Mat& frame_current, const cv::Mat& frame_previous, cv::Mat& score, double rho)
{

    cv::Mat frame_current_gradx, frame_current_grady ;
    cv::Mat frame_previous_gradx, frame_previous_grady;
    cv::Mat gradx, grady, gradt;
    cv::Mat J11, J12, J13, J22, J23, J33, H;
    Mat I1, I2, I3;



    // compute gradients
    Gradient(frame_current, frame_current_gradx, frame_current_grady );
    Gradient(frame_previous, frame_previous_gradx, frame_previous_grady);
    gradt = frame_current-frame_previous;
    gradx = (frame_current_gradx+frame_previous_gradx)/2.0;
    grady = (frame_current_grady+frame_previous_grady)/2.0;




    // motion tensor smoothing, only for spatial scales
    cv::GaussianBlur(gradx.mul(gradx), J11, cv::Size(0,0),rho, rho );
    cv::GaussianBlur(gradx.mul(grady), J12, cv::Size(0,0),rho, rho );
    cv::GaussianBlur(gradx.mul(gradt), J13, cv::Size(0,0),rho, rho );
    cv::GaussianBlur(grady.mul(grady), J22, cv::Size(0,0),rho, rho );
    cv::GaussianBlur(grady.mul(gradt), J23, cv::Size(0,0), rho, rho);
    cv::GaussianBlur(gradt.mul(gradt), J33, cv::Size(0,0), rho, rho);


    // tensor invariants
    I1 = J11+J22+J33;
    I2 = J11.mul(J22) + J11.mul(J33) + J22.mul(J33) - J12.mul(J12) - J13.mul(J13) - J23.mul(J23);
    I3 = (J11.mul(J22)).mul(J33) + 2*(J12.mul(J23)).mul(J13)
    - (J13.mul(J13)).mul(J22) - (J12.mul(J12)).mul(J33)
    - (J11.mul(J23)).mul(J23);


    switch(_score_method)
    {
    case Harris:
        score = I3 - 0.005*((I1).mul(I1)).mul(I1);
        break;
    case MinEigen:
        /// This case uses two frames. Instead of following Laptev's method, the score is evaluated by the minimal eigenvalue of
        /// the motion tensor. This can be regarded as a spatial-temporal 'good feature for tracking'.
        /// Here the non-iterative method of K.M. Hasan et al. is used.
        /// [1] Analytical Computation of the Eigenvalues and Eigenvectors in DT-MRI
        double v, s, phi;
        int i,j;
        score.create(I1.rows, I1.cols, CV_64F);

        for (j = 0; j < J11.rows; j++)
        {
            double* ptr_I1 = (double*) I1.ptr(j);
            double* ptr_I2 = (double*) I2.ptr(j);
            double* ptr_I3 = (double*) I3.ptr(j);
            double* ptr_score = (double*) score.ptr(j);

            for(i = 0; i < J11.cols; i++)
            {
                v = (ptr_I1[i]/3.0)*(ptr_I1[i]/3.0) - ptr_I2[i]/3.0;

                s = (ptr_I1[i]/3.0)*(ptr_I1[i]/3.0)*(ptr_I1[i]/3.0)-ptr_I1[i]*ptr_I2[i]/6.0+ptr_I3[i]/2.0;

                phi = std::acos(s/v*std::sqrt(1.0/v));

                ptr_score[i] = ptr_I1[i]/3 - 2*std::sqrt(v) * std::cos(PI/3.0-phi);
            }
        }

    default:
        break;
    }


}



void StipDetector::VideoKeypointDisplay( )
{
    Mat frame = (_frame_current + _frame_previous)/2.0;
    namedWindow("results",WINDOW_NORMAL);
    frame.convertTo(frame, CV_8U);

    cv::Mat frame_display;

    cv::drawKeypoints(frame, _corners, frame_display, Scalar::all(-1),4);

    imshow("results", frame_display);
    waitKey(30);
}



StipDetector::StipDetector() : _n_level(1), _scale_base(0.0)
{
    _corners.clear();

}

StipDetector::StipDetector(const Mat& f1, const Mat& f2) : _scale_base(1.5),_n_level(3)
{
    _frame_current = f2.clone();
    _frame_previous = f1.clone();
    _corners.clear();
    _roi_method = StipDetector::ScoreThreshold;
    _score_method = StipDetector::Harris;

}

void StipDetector::SetFrames(const cv::Mat& f1, const cv::Mat& f2)
{
    _frame_current = f2.clone();
    _frame_previous = f1.clone();
}

void StipDetector::SetFineScale(double scale)
{
    _scale_base = scale;
}



void StipDetector::SetLevelNum( int n_level)
{
    _n_level = n_level;
}


void StipDetector::SetMethodScore(StipDetector::ScoreMethod method)
{
    _score_method = method;
}


void StipDetector::SetMethodROI(StipDetector::RoiMethod method)
{
    _roi_method = method;
    if(method == StipDetector::GM2)
        _pMOG= cv::createBackgroundSubtractorMOG2(1000);


}

void StipDetector::ClearPoints()
{
    _corners.clear();
}

void StipDetector::DefineROI()
{
    double min_val, max_val;
    switch(_roi_method)
    {
    case ScoreThreshold:
        _roi = abs(_score);
        minMaxLoc(_roi,&min_val,&max_val);
        _roi -= min_val;
        convertScaleAbs(_roi,_roi,255/(max_val-min_val));
        cv::threshold(_roi, _roi,10,255,cv::THRESH_BINARY);
        _roi.convertTo(_roi,CV_32F);
        break;

    case GM2:
        _pMOG->apply(_frame_current, _roi);
        StipDetector::Close(_roi,_roi,3);
        _roi.convertTo(_roi,CV_32F);
        break;

    default:
        break;


    }
}

void StipDetector::GetROI(cv::Mat& fg)
{
    fg = _roi.clone();
}

void StipDetector::GetScore(cv::Mat& score)
{
    score = _score.clone();
}

void StipDetector::GetKeyPoints(vector<cv::KeyPoint>& corners)
{
    corners = _corners;
}



void StipDetector::detect()
{
    double min_val, max_val;
    const double ss_multiplier = 2;
;
    int i,j;

    /// scale-space representation, rho_s = pow(ss_multiplier,s)*rho, s = 0,...,_n_level-1
    for (int s = 0; s < _n_level; s++)
    {
        StipDetector::MotionTensorScore (_frame_current, _frame_previous, _score, _scale_base*std::pow(ss_multiplier,s));

        // find local maximum
        StipDetector::Dilation(_score, _score_dilate,11);
        _score_peak = _score_dilate-_score;
        _score_peak.convertTo(_score_peak, CV_32F);
        //cv::threshold(score_peak, score_peak, 0,255.0,cv::THRESH_BINARY );

        StipDetector::DefineROI();


        // extract key points
        for (j = 0; j < _score.rows; j++)
        {

            float* ptr_roi = (float*) _roi.ptr(j);
            float* ptr_score_peak = (float*) _score_peak.ptr(j);
            for (i = 0; i < _score.cols; i++)
            {
                if(ptr_roi[i] != 0.0f && ptr_score_peak[i] == 0.0f)
                    _corners.push_back(KeyPoint(Point2f(i,j),std::pow(ss_multiplier,s)*_scale_base) );

            }
        }


    }

}



StipDetector::~StipDetector( )
{
    //dtor
}
