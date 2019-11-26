#include<iostream>
#include<algorithm>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;

int saturation;	// ���Ͷ�
const int max_increment = 200;
Mat img, new_img;	   //img:ԭʼͼ��; new_img:����Ҫչʾͼ��;

void saturability (int, void*);

//�������Ͷ�
void saturability (int, void*)
{
	float increment = (saturation - 80) * 1.0 / max_increment;
	for (int col = 0; col < img.cols; col++)
	{
		for (int row = 0; row < img.rows; row++)
		{
			// R,G,B �ֱ��Ӧ�������±�� 2,1,0
			uchar r = img.at<Vec3b> (row, col)[2];
			uchar g = img.at<Vec3b> (row, col)[1];
			uchar b = img.at<Vec3b> (row, col)[0];

			float maxn = max (r, max (g, b));
			float minn = min (r, min (g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // ��Ϊ 0 �����Ͷ�Ϊ0�����ܵ��ڱ��Ͷȣ�����ԭ���ص�
			{
				new_img.at<Vec3b> (row, col)[0] = b;
				new_img.at<Vec3b> (row, col)[1] = g;
				new_img.at<Vec3b> (row, col)[2] = r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)		// ��������ֵ�������Ͷȣ���[0, 1]��Χ��
				sat = delta / value;
			else
				sat = delta / (2 - value);

			// ���о���ı��Ͷȵ���
			if (increment >= 0)		// �������� 0�����Ͷȳʼ����������������Լ�˥
			{
				if ((increment + sat) >= 1)	// ���� + ���Ͷȴ��� 1���������Ͷȵ�����
					alpha = sat;
				else	// ����ȡ������ 1 �Ĳ���
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;	// �󵼣�ʵ�ּ�����ǿ
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
	img = imread ("test1.jpg");		// ����ͼƬ�������� Mat ���� img ��
	new_img = Mat::zeros (img.size (), img.type ());    // ����Ҫչʾ����Ķ���

	saturation = 10;	//���Ͷȳ�ʼֵ

	namedWindow ("Effect Image", WINDOW_NORMAL);		// ����Ч��ͼ����

	createTrackbar ("Saturability:", "Effect Image", &saturation, 200, saturability); // �������ͶȻ�����

	// �����ص�����Ϊ��ȫ�ֱ��������� userdata Ϊ0
	saturability (saturation, 0);

	cv::waitKey (0);
	return 0;
}

