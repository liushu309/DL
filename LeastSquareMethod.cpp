#include <opencv2/opencv.hpp>
#include <iostream>

/*
最小二乘法的实现
C++版
命令行输入数据文件
最后输入x得到预测的y值
数据文件格式如下
data.txt
1 0 
2 1 
3 2 
0 1 
1 2 
2 3

结果应为 
y = 0.818182+0.454545x

*/
#include<iostream>
#include<fstream>
#include<vector>
using namespace std;

class LeastSquare {
    double b0, b1;
public:
    LeastSquare(const vector<double>& x, const vector<double>& y)
    {
        double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
        for (int i = 0; i<x.size(); ++i)
        {
            t1 += x[i] * x[i];
            t2 += x[i];
            t3 += x[i] * y[i];
            t4 += y[i];
        }

        b0 = (t1*t4 - t2*t3) / (t1*x.size() - t2*t2);        // 求得 B0
        b1 = (t3*x.size() - t2*t4) / (t1*x.size() - t2*t2);  // 求得 B1 
    }

    double getY(const double x) const
    {
        return b0+b1*x;
    }

    void print() const
    {
        if (b1>=0)
            cout << "y = " << b0 << "+" << b1 << 'x' << "\n";
        else
            cout << "y = " << b0 << "" << b1 << 'x' << "\n";
    }

};

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << " data.txt don't exit " << endl;
        return -1;
    }
    else
    {
        vector<double> x;
        vector<double> y;
        int count = 1;
        ifstream in(argv[1]);
        for (double d; in >> d; count++)
            if (count % 2 == 1)
                x.push_back(d);
            else
                y.push_back(d);
        LeastSquare ls(x, y);
        ls.print();

        cout << "Input x:\n";
        double x0;
        while (cin >> x0)
        {
            cout << "y = " << ls.getY(x0) << endl;
            cout << "Input x:\n";
        }
    }
    int endline;
    cin >> endline;
}

