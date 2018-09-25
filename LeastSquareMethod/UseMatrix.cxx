/*
直接调用矩阵公式版
*/

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    cv::Mat A(6, 2, CV_32FC1);
    A.at<float>(0, 0) = 1;
    A.at<float>(0, 1) = 1;
    A.at<float>(1, 0) = 2;
    A.at<float>(1, 1) = 1;
    A.at<float>(2, 0) = 3;
    A.at<float>(2, 1) = 1;
    A.at<float>(3, 0) = 0;
    A.at<float>(3, 1) = 1;
    A.at<float>(4, 0) = 1;
    A.at<float>(4, 1) = 1;
    A.at<float>(5, 0) = 2;
    A.at<float>(5, 1) = 1;
    
    std::cout << "theA is:\n" << A << std::endl;

    cv::Mat B(6, 1, CV_32FC1);
    B.at<float>(0, 0) = 0;
    B.at<float>(1, 0) = 1;
    B.at<float>(2, 0) = 2;
    B.at<float>(3, 0) = 1;
    B.at<float>(4, 0) = 2;
    B.at<float>(5, 0) = 3;

    std::cout<<"the B is:\n" << B << std::endl;

    cv::Mat res;
    res = (A.t() * A).inv() * A.t() * B;
    std::cout << "the res is \n" << res << std::endl;

    return 0;
}
