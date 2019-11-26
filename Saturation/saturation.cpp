#include<iostream>
#include<algorithm>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;

int saturation;	// 饱和度
const int max_increment = 200;
Mat img, new_img;	   //img:原始图像; new_img:最终要展示图像;

void saturability (int, void*);

//调整饱和度
void saturability (int, void*)
{
	float increment = (saturation - 80) * 1.0 / max_increment;
	for (int col = 0; col < img.cols; col++)
	{
		for (int row = 0; row < img.rows; row++)
		{
			// R,G,B 分别对应数组中下标的 2,1,0
			uchar r = img.at<Vec3b> (row, col)[2];
			uchar g = img.at<Vec3b> (row, col)[1];
			uchar b = img.at<Vec3b> (row, col)[0];

			float maxn = max (r, max (g, b));
			float minn = min (r, min (g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // 差为 0 即饱和度为0，不能调节饱和度，保存原像素点
			{
				new_img.at<Vec3b> (row, col)[0] = b;
				new_img.at<Vec3b> (row, col)[1] = g;
				new_img.at<Vec3b> (row, col)[2] = r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)		// 根据亮度值调整饱和度，在[0, 1]范围内
				sat = delta / value;
			else
				sat = delta / (2 - value);

			// 进行具体的饱和度调整
			if (increment >= 0)		// 增量大于 0，饱和度呈级数增长，否则线性减衰
			{
				if ((increment + sat) >= 1)	// 增量 + 饱和度大于 1，超过饱和度的上限
					alpha = sat;
				else	// 否则，取增量对 1 的补数
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;	// 求导，实现级数增强
				new_r = r + (r - light * 255) * alpha;
				new_g = g + (g - light * 255) * alpha;
				new_b = b + (b - light * 255) * alpha;
			}
			else
			{
				alpha = increment;
				new_r = light * 255 + (r - light * 255) * (1 + alpha);
				new_g = light * 255 + (g - light * 255) * (1 + alpha);
				new_b = light * 255 + (b - light * 255) * (1 + alpha);
			}
			new_img.at<Vec3b> (row, col)[0] = new_b;
			new_img.at<Vec3b> (row, col)[1] = new_g;
			new_img.at<Vec3b> (row, col)[2] = new_r;
		}
	}
	imshow ("Effect Image", new_img);
}

int main ()
{
	img = imread ("test1.jpg");		// 加载图片，保存在 Mat 对象 img 中
	new_img = Mat::zeros (img.size (), img.type ());    // 最终要展示结果的对象

	saturation = 10;	//饱和度初始值

	namedWindow ("Effect Image", WINDOW_NORMAL);		// 创建效果图窗口

	createTrackbar ("Saturability:", "Effect Image", &saturation, 200, saturability); // 创建饱和度滑动条

	// 函数回调，因为是全局变量，所以 userdata 为0
	saturability (saturation, 0);

	cv::waitKey (0);
	return 0;
}

