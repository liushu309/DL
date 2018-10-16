#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace  cv;
//SVD算法
int main()
{
	string filename = "lenna.jpg";
	Mat img = imread(filename, IMREAD_GRAYSCALE);
	img.convertTo(img, CV_32FC1);//转换为float类型
	string tile = "压缩比率";
	Mat U, W, VT;
	SVD svd;
	svd.compute(img, W, U, VT);
    std::cout<<"w size is: " << W.size()<<std::endl;
    std::cout<<"U size is: " << U.size() <<std::endl;
    std::cout<< "VT size is: " << VT.size() << std::endl;
    std::cout << W;
	//将矩阵进行压缩
	//原始数据进行压缩之后是 m*m m*n n*n
	//由于特征值在前几行中占有的比率是比较大的，所以，仅仅选择前几行(r)作为好的特征值
	//那么分解的结果就变为 m*r r*r r*n 这样的形式
	double radio = 0.2;//压缩比率
	int rows = radio*img.rows;
	Mat WROI = Mat::zeros(rows, rows, CV_32FC1);//W矩阵的大小
	//填充举着WROI;
	for (int i = 0; i < rows;++i)
	{
		WROI.at<float>(i, i) = W.at<float>(i);
	}
	Mat UROI = U.colRange(0, rows);//主要注意的是，colrange中不包含End
	Mat VTROI = VT.rowRange(0, rows);
	Mat Result = UROI*WROI*VTROI;
	//将结果转换为灰度影像
	Result.convertTo(Result, CV_8UC1);
	namedWindow(tile);
	imshow(tile, Result);
	waitKey(0);
	return 0;
}
