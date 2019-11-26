#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2\imgproc\types_c.h>
#include<iostream>
#include<algorithm>
#include<cmath>
using namespace std;
using namespace cv;

const static double pi = 3.1415926;
constexpr auto WRONG = -1;
Mat dst;

void GaussianFilter (Mat src, int kernel, double sigma)
{
	int ksize = 2 * kernel + 1;
	// ������άȨ�ؾ���
	double** templateMatrix = new double* [ksize];
	for (int i = 0; i < ksize; i++) {
		templateMatrix[i] = new double[ksize];
	}

	double x2, y2;
	double sum = 0.0;
	//�������˸��������Ȩ��ֵ,ֻ�ü���һ��
	for (int i = 0; i < ksize; i++)
	{
		x2 = pow (i - kernel, 2);
		for (int j = 0; j < ksize; j++)
		{
			y2 = pow (j - kernel, 2);
			double g = exp (-(x2 + y2) / (2 * sigma * sigma));
			sum += g;
			templateMatrix[i][j] = g;
		}
	}
	// ����ĵ�����Ȩ��
	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			templateMatrix[i][j] /= sum;
		}
	}

	// Ȩ��Ӧ�õ�ͼ����
	int row = src.rows;
	int col = src.cols;

	dst = Mat::zeros (src.rows, src.cols, src.type ());

	//ֱ�Ӵӡ��м䡱��ʼ��ʡȥǰ�����ҵı���
	for (int x = 0; x < row; x++)
	{
		for (int y = 0; y < col; y++)
		{
			for (int c = 0; c < 3; c++)
			{
				if (x >= kernel && x <= row - kernel - 1 && y >= kernel && y <= col - kernel - 1)
				{
					double sum = 0.0;
					for (int i = -kernel; i <= kernel; i++)
					{
						for (int j = -kernel; j <= kernel; j++)
						{
							//�� templateMatrix ��Զ�ھ�������귶Χ�ڱ���
							sum += src.at<Vec3b> (x + i, y + j)[c] * templateMatrix[i + kernel][j + kernel];
						}
					}
					if (sum > 255.0) {
						sum = 255;
					}
					if (sum < 0.0) {
						sum = 0;
					}
					dst.at<Vec3b> (x, y)[c] = sum;
				}
				else //��Ե����ֱ�����
				{
					dst.at<Vec3b> (x, y)[c] = src.at<Vec3b> (x, y)[c];
				}
			}
		}
	}
}

static Mat src;
static int kernel = 0;
static double sigma;  // С��sigma��ʵ��ʹ��
static int max_kernel = 10;
static int intsigma = 15;	// ����sigma��Ϊ�˼Ӵ󻬶���
static int max_intsigma = 50;

static void callBackAdjust (int, void*)
{
	sigma = 0.05 * intsigma;
	GaussianFilter (src, kernel, sigma);
	imshow ("Final Image", dst);
}

int main ()
{
	src = imread ("lena.jpg");
	if (!src.data) {
		cout << "error read image" << endl;
		return WRONG;
	}

	namedWindow ("Final Image", WINDOW_NORMAL);

	createTrackbar ("kernel", "Final Image", &kernel, max_kernel, callBackAdjust);
	createTrackbar ("sigma", "Final Image", &intsigma, max_intsigma, callBackAdjust);
	callBackAdjust (0, 0);

	waitKey (0);
	return 0;
}