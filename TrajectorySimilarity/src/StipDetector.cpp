#include "StipDetector.h"




void StipDetector::MotionTensorScore (const cv::Mat& frame_current, const cv::Mat& frame_previous, cv::Mat& score, double rho)
{

    cv::Mat frame_current_gradx, frame_current_grady ;
    cv::Mat frame_previous_gradx, frame_previous_grady;
    cv::Mat gradx, grady, gradt;
    cv::Mat J11, J12, J13, J22, J23, J33, H;
    Mat trace_map, det_map;


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


    trace_map = J11+J22+J33;
    det_map = (J11.mul(J22)).mul(J33) + 2*(J12.mul(J23)).mul(J13)
    - (J13.mul(J13)).mul(J22) - (J12.mul(J12)).mul(J33)
    - (J11.mul(J23)).mul(J23);



    score = det_map - 0.005*((trace_map).mul(trace_map)).mul(trace_map);

}


void StipDetector::Dilation(const cv::Mat& src, cv::Mat& dst, int kernelsize=5)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation
    cv::dilate( src, dst, element );

}





void StipDetector::Gradient (const Mat& src, Mat& gradx, Mat& grady, bool use_sobel = true)
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
    frame_current = Mat::eye(2,2,CV_64F);
    frame_previous = Mat::eye(2,2,CV_64F);
    _corner.clear();


}

StipDetector::StipDetector(const Mat& f1, const Mat& f2) : _scale_base(2.5),_n_level(3)
{
    _frame_current = f2.clone();
    _frame_previous = f1.clone();
    _corner.clear();

}




StipDetector::StipDetector(const Mat& f1, const Mat& f2, double scale_fine, int n_level) : _scale_base(scale_fine),_n_level(n_level)
{
    _frame_current = f2.clone();
    _frame_previous = f1.clone();

    _scale_base= scale_fine;
    _n_level = n_level;

    _corner.clear();

}

void StipDetector::SetFineScale(double scale)
{
    _scale_base = scale;
}

void StipDetector::SetLevelNum( int n_level)
{
    _n_level = n_level;
}



void StipDetector::detect()
{
    double min_val, max_val;
    const double ss_multiplier = 1.6;
    cv::Mat _score, _score_roi, _score_peak, _score_dilate;


    /// scale-space representation, rho_s = pow(1.6,s)*rho, s = 0,...,_n_level-1
    for (int s = 0; s < _n_level; s++)
    {
        MotionTensorScore (_frame_current, _frame_previous, _score, _scale_base*std::pow(ss_multiplier,s));

        // find local maximum
        Dilation(_score, _score_dilate,11);
        _score_peak = _score_dilate-_score;
        _score_peak.convertTo(_score_peak, CV_32F);
        //cv::threshold(score_peak, score_peak, 0,255.0,cv::THRESH_BINARY );

        // find ROI
        _score_roi = abs(_score);
        minMaxLoc(_score_roi,&min_val,&max_val);
        _score_roi -= min_val;
        convertScaleAbs(_score_roi,_score_roi,255/(max_val-min_val));
        cv::threshold(_score_roi, _score_roi,10,255,cv::THRESH_BINARY_INV);
        score_roi.convertTo(_score_roi,CV_32F);

        // extract key points
        for (j = 0; j < _score.rows; j++)
        {

            float* ptr_score_roi = (float*) _score_roi.ptr(j);
            float* ptr_score_peak = (float*) _score_peak.ptr(j);
            for (i = 0; i < _score.cols; i++)
            {
                if(ptr_score_roi[i] + ptr_score_peak[i] == 0.0f)
                    corners.push_back(KeyPoint(Point2f(i,j),std::pow(ss_multiplier,s)*_scale) );

            }
        }


    }


}



StipDetector::~StipDetector( )
{
    //dtor
}
