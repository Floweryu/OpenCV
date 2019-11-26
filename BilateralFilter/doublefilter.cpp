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

void BilateralFilter (Mat src, int kernel, const int space_sigma, const int color_sigma)
{
	int ksize = 2 * kernel + 1;
	// 创建二维权重矩阵
	double** templateMatrix = new double* [ksize];
	for (int i = 0; i < ksize; i++) {
		templateMatrix[i] = new double[ksize];
	}

	double x2, y2;
	double sum = 0.0;
	//计算卷积核各个坐标的权重值,只用计算一次
	for (int i = 0; i < ksize; i++)
	{
		x2 = pow (i - kernel, 2);
		for (int j = 0; j < ksize; j++)
		{
			y2 = pow (j - kernel, 2);
			double g = exp (-(x2 + y2) / (2 * space_sigma * space_sigma));
			sum += g;
			templateMatrix[i][j] = g;
		}
	}
	// 计算的到最终权重
	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			templateMatrix[i][j] /= sum;
		}
	}


	// 创建颜色权重
	double* color_weight = new double[256];
	double color_coeff = -0.5 / (color_sigma * color_sigma);
	for (int i = 0; i < 256; i++)
		color_weight[i] = exp (i * i * color_coeff);


	int row = src.rows;
	int col = src.cols;

	dst = Mat::zeros (src.rows, src.cols, src.type ());

	for (int x = 0; x < row; x++)
	{
		for (int y = 0; y < col; y++)
		{
			for (int c = 0; c < 3; c++)
			{
				// 去掉四周
				if (x >= kernel && x <= row - kernel - 1 && y >= kernel && y <= col - kernel - 1)
				{
					double weight_sum = 0.0, pix_sum = 0.0;
					for (int i = -kernel; i <= kernel; i++)
					{
						for (int j = -kernel; j <= kernel; j++)
						{
							uchar center = src.at<Vec3b> (x, y)[c];
							uchar around_center = src.at<Vec3b> (x + j, y + j)[c];
							double weight = templateMatrix[i + kernel][j + kernel] * color_weight[abs(center - around_center)];
							// 每个像素点的颜色值乘以权重相加
							weight_sum += weight;
							pix_sum += src.at<Vec3b> (x + i, y + j)[c] * weight;
						}
					}
					dst.at<Vec3b> (x, y)[c] = pix_sum / weight_sum;
				}
				else //边缘地区直接相等
				{
					dst.at<Vec3b> (x, y)[c] = src.at<Vec3b> (x, y)[c];
				}
			}
		}
	}
}

static Mat src;
static int kernel = 0;
static int max_kernel = 5;
static int ksize;
static int sigma = 10;
static int space_sigma = 10;
static int color_sigma = 10;
static int max_intsigma = 100;

static void callBackAdjust (int, void*)
{
	ksize = 2 * kernel + 1;
	BilateralFilter (src, ksize, space_sigma, color_sigma);
	imshow ("Final Image", dst);
}

int main ()
{
	src = imread ("lena.jpg");
	if (!src.data) {
		cout << "error read image" << endl;
		return WRONG;
	}

	namedWindow ("Final Image", WINDOW_AUTOSIZE);

	createTrackbar ("kernel", "Final Image", &kernel, max_kernel, callBackAdjust);
	createTrackbar ("SpaceSigma", "Final Image", &space_sigma, max_intsigma, callBackAdjust);
	createTrackbar ("ColorSigma", "Final Image", &color_sigma, max_intsigma, callBackAdjust);
	callBackAdjust (0, 0);

	waitKey (0);
	return 0;
}