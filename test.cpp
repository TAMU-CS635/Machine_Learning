#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <string.h>

using namespace cv;
using namespace std;

int test_sample = 10;

int main()
{
    int img_area = 320*240;
    Mat img_mat;
    int ii = 0;

    CvSVM svm;
    
    svm.load("firstrial");
    
    chdir("/Users/khuongnguyen/Desktop/New_SVM/Test");
    for(int n = 0; n < test_sample; n++){
        const string file = to_string(n) + ".jpg";
        img_mat = imread(file,1); // I used 0 for greyscale
        cvtColor( img_mat, img_mat, CV_BGR2GRAY );
        GaussianBlur(img_mat, img_mat, Size(7,7), 1.5, 1.5);
        Canny(img_mat, img_mat, 0, 30, 3);
        resize(img_mat, img_mat, Size(320, 240), 0, 0, INTER_LINEAR);
        Mat test(1,img_area,CV_32FC1);
    
        ii = 0; // Current column in training_mat
        for (int i = 0; i<img_mat.rows; i++) {
            for (int j = 0; j < img_mat.cols; j++) {
                test.at<float>(0,ii++) = img_mat.at<uchar>(i,j);
            }
        }
        int testid = svm.predict(test);
        
        switch (testid) {
            case 0:
                std::cout << file << " is on the left : " << testid << std::endl;
                break;
            case 1:
                std::cout << file << " is on the left_mid : " << testid << std::endl;
                break;
            case 2:
                std::cout << file << " is in the middle : " << testid << std::endl;
                break;
            case 3:
                std::cout << file << " is on the right_mid : " << testid << std::endl;
                break;
            case 4:
                std::cout << file << " is on the right : " << testid << std::endl;
                break;
            default:
                break;
        }
        
    }
    
    return 0;
}
