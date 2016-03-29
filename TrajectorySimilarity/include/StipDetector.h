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
        virtual ~StipDetector();
    protected:
    private:
        int i,j,t;
        vector<Keypont> corners;


};

#endif // STIPDETECTOR_H
