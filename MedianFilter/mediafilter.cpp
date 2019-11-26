#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2\imgproc\types_c.h>
#include<iostream>
#include<algorithm>
using namespace std;
using namespace cv;

constexpr auto WRONG = -1;
Mat dst;

uchar GetMedian (uchar tar[], int length)
{
	sort (tar, tar + length);
	return tar[length / 2];
}

void AverFilter (Mat src, int kernel)
{
	int row = src.rows;
	int col = src.cols;
	int block = 2 * kernel + 1;
	dst = Mat::zeros (src.rows, src.cols, src.type ());
	uchar* arr = new uchar[block * block + 1];	// 存储所选块的数值

	for (int x = 0; x < row; x++)
	{
		for (int y = 0; y < col; y++)
		{
			for (int c = 0; c < 3; c++)
			{
				int ans = 0;  
				if (x >= kernel && x <= row - kernel - 1 && y >= kernel && y <= col - kernel - 1)
				{
					for (int i = x - kernel; i <= x + kernel; i++)
					{
						for (int j = y - kernel; j <= y + kernel; j++)
						{
							arr[ans++] =  src.at<Vec3b> (i, j)[c];
						}
					}
					uchar median = GetMedian (arr, block * block);
					dst.at<Vec3b> (x, y)[c] = median;
				}
				else {
					dst.at<Vec3b> (x, y)[c] = src.at<Vec3b> (x, y)[c];
				}
			}
		}
	}
}

static Mat src;
static int kernel = 0;
static int max_kernel = 10;

static void callBackAdjust (int, void*)
{
	AverFilter (src, kernel);
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
	callBackAdjust (0, 0);

	waitKey (0);
	return 0;
}