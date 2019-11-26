#define  _USE_MATH_DEFINES
#include<iostream>
#include<algorithm>
#include<cmath>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;

constexpr auto WRONG = -1;
#define CLIP_RANGE(value, min, max) ((value > max) ? max : (value < min) ? min : value)
#define COLOR_RANGE(value) CLIP_RANGE(value, 0, 255)

// 调整对比度和亮度
int ContrastAndBright (InputArray src, OutputArray dst, int brightValue, int contrastValue)
{
	Mat input = src.getMat ();
	if (input.empty ()) {
		return WRONG;
	}

	dst.create (src.size (), src.type ());
	Mat output = dst.getMat ();


	brightValue = CLIP_RANGE (brightValue, -255, 255);
	contrastValue = CLIP_RANGE (contrastValue, -255, 255);

	double brightness = brightValue / 255.;
	double contrast = contrastValue / 255.;
	double k = tan ((45 + 44 * contrast) / 180 * M_PI);

	Mat lookupTable (1, 256, CV_8U);
	uchar* p = lookupTable.data;
	for (int i = 0; i < 256; i++)
		p[i] = COLOR_RANGE ((i - 127.5 * (1 - brightness)) * k + 127.5 * (1 + brightness));

	LUT (input, lookupTable, output);

	return 0;
}

static Mat src;
static int brightValue = 255;
static int contrastValue = 255;
static int max_brightValue = 470;
static int max_contrastValue = 450;

static void callbackAdjust (int, void*)
{
	Mat dst;
	ContrastAndBright (src, dst, brightValue - 255, contrastValue - 245);
	imshow ("Effect Image", dst);
}

int main ()
{
	src = imread ("test1.jpg");		// 加载图片，保存在 Mat 对象 img 中
	if (!src.data) {
		cout << "error read image" << endl;
		return WRONG;
	}

	namedWindow ("Effect Image", WINDOW_NORMAL);		// 创建效果图窗口

	createTrackbar ("Contrast:", "Effect Image", &contrastValue, max_contrastValue, callbackAdjust);	// 创建对比度滑动条
	createTrackbar ("Brightness:", "Effect Image", &brightValue, max_brightValue, callbackAdjust);	// 创建亮度滑动条

	callbackAdjust (0, 0);

	cv::waitKey (0);
	return 0;
}



