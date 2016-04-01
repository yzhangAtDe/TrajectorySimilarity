#include<iostream>
#include<opencv/cv.hpp>
#include<opencv/highgui.h>

#define PI 3.1415926

using namespace std;
using namespace cv;


void DoubleMatDisplay(const Mat& src, const string window_name)
{
    double min_val, max_val;
    Mat display;

    minMaxLoc(src, &min_val, &max_val);
    src -= min_val;

    cout << "variable=" << window_name << " " << "minimum= " << min_val << " "<<"maximum=" << max_val <<endl;

    src.convertTo(display,CV_8U,255.0/(max_val-min_val));

    imshow(window_name,display);
    waitKey(0);
}


void VideoKeypointDisplay( Mat& frame, const vector<KeyPoint>& corners)
{

    namedWindow("results",WINDOW_NORMAL);
    frame.convertTo(frame, CV_8U);

    cv::Mat frame_display;

    cv::drawKeypoints(frame, corners, frame_display, Scalar::all(-1),4);

    imshow("results", frame_display);
    waitKey(30);
}



void Gradient (const Mat& src, Mat& gradx, Mat& grady, bool use_sobel = true)
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


void MotionTensorScore1 (const cv::Mat& frame_current, const cv::Mat& frame_previous, const cv::Mat& frame_previous2,cv::Mat& score, double rho)
/// this method applies three frames instead of two. It averages the temporal derivatives so as to eliminate the influence of noise.
{

    cv::Mat frame_current_gradx, frame_current_grady ;
    cv::Mat frame_previous_gradx, frame_previous_grady;
    cv::Mat frame_previous2_gradx, frame_previous2_grady;

    cv::Mat gradx, grady, gradt;
    cv::Mat J11, J12, J13, J22, J23, J33, H;
    Mat trace_map, det_map;
    cv::Mat eigenvalue;


    // compute gradients
    Gradient(frame_current, frame_current_gradx, frame_current_grady );
    Gradient(frame_previous, frame_previous_gradx, frame_previous_grady);
    Gradient(frame_previous2, frame_previous2_gradx, frame_previous2_grady);

    gradt = (frame_current+frame_previous)/2.0 - (frame_previous2+frame_previous)/2.0;
    gradx = (frame_current_gradx+frame_previous_gradx + frame_previous2_gradx)/3.0;
    grady = (frame_current_grady+frame_previous_grady + frame_previous2_gradx)/3.0;




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

//    DoubleMatDisplay(J11, "J11");
//    DoubleMatDisplay(J12, "J12");
//    DoubleMatDisplay(J13, "J13");
//    DoubleMatDisplay(J22, "J22");
//    DoubleMatDisplay(J23, "J23");
//    DoubleMatDisplay(J33, "J33");

//    DoubleMatDisplay(frame_current, "frame_current");
//
//    DoubleMatDisplay(det_map, "det");



    score = det_map - 0.005*((trace_map).mul(trace_map)).mul(trace_map);

}



void MotionTensorScore2 (const cv::Mat& frame_current, const cv::Mat& frame_previous,cv::Mat& score, double rho)
/// This function uses two frames. Instead of following Laptev's method, the score is evaluated by the minimal eigenvalue of
/// the motion tensor. This can be regarded as a spatial-temporal 'good feature for tracking'.
/// Here the non-iterative method of K.M. Hasan et al. is used.
/// [1] Analytical Computation of the Eigenvalues and Eigenvectors in DT-MRI


{

    cv::Mat frame_current_gradx, frame_current_grady ;
    cv::Mat frame_previous_gradx, frame_previous_grady;

    cv::Mat gradx, grady, gradt;
    cv::Mat J11, J12, J13, J22, J23, J33, H;
    Mat I1, I2, I3;
    cv::Mat eigenvalue;


    // compute gradients
    Gradient(frame_current, frame_current_gradx, frame_current_grady );
    Gradient(frame_previous, frame_previous_gradx, frame_previous_grady);

    gradt = (frame_current-frame_previous)/2.0 ;
    gradx = (frame_current_gradx+frame_previous_gradx )/2.0;
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


}




void Dilation(const cv::Mat& src, cv::Mat& dst, int kernelsize=5)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation
    cv::dilate( src, dst, element );

}






int main (int argc, char** argv)
{

    // argument transfering
    if(argc != 4)
    {
        std::cout << "wrong arguments number."<< std::endl;
        return -1;
    }
    char* filename = argv[1];
    double sigma = atof(argv[2]);
    double rho = atof(argv[3]);


    // video IO
    cv::VideoCapture cap(filename);



    if(!cap.isOpened())
    {
        std::cout<< "video is not opened!"<< std::endl;
        return -2;
    }

    // variable configuration
    std::vector<cv::KeyPoint> corners;
    cv::Mat frame,frame_current, frame_previous, frame_previous2;
    cv::Mat score, score_roi, score_peak, score_dilate;

    int t=0;
    int i,j,s;
    double min_val, max_val;

    Mat fgMaskMOG2;
    Ptr<BackgroundSubtractor> pMOG2;
    pMOG2 = createBackgroundSubtractorMOG2();




    while(1)
    {
        // release the corner vector
        corners.clear();

        // read current frame from the video
        cap >> frame;
        cv::Size size(200,150);
        cv::resize(frame,frame,size);

        //Gamma correction
        frame.convertTo(frame,CV_32F);
        cv::Ptr<Tonemap> map = cv::createTonemap(2.2f);
        map -> process(frame,frame);
        frame = frame*255;

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


        int s_max=5;
        if(t>1)
        {


            /// scale-space representation, rho_s = pow(1.5,s)*rho, s = 1,2,3,...,5
            for (s = 0; s < s_max; s++)
            {
//                MotionTensorScore2 (frame_current, frame_previous,score, rho*std::pow(2,s));
//
//                // find local maximum
//                Dilation(score, score_dilate,7);
//                score_peak = score_dilate-score;
//                score_peak.convertTo(score_peak, CV_32F);
                //cv::threshold(score_peak, score_peak, 0,255.0,cv::THRESH_BINARY );

                // find ROI
//                score_roi = abs(score);
//            //                cv::exp(score_roi, score_roi);
//                minMaxLoc(score_roi,&min_val,&max_val);
//                score_roi = 255* (score_roi-min_val)/(max_val-min_val);
//                score_roi.convertTo(score_roi,CV_32F);
//                cv::threshold(score_roi, score_roi,50,255,cv::THRESH_BINARY_INV);

                pMOG2 ->apply(frame_current, fgMaskMOG2);

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
            imshow("segmentation",fgMaskMOG2);

            VideoKeypointDisplay(frame_middle, corners);

        }



        // for the next frame
        frame_previous.copyTo(frame_previous2);
        frame_current.copyTo(frame_previous);

        t++;

    }




    return 0;

}
