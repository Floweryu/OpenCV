#include<opencv2/opencv.hpp>
#include<vector>
#include<opencv2/xfeatures2d.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main ()
{
	Mat img_l, img_r, img_grayl, img_grayr;
	Mat LRmatcher;
	img_l = imread ("D:\\Learn_Files\\OpenCV\\StitchDetial\\left.jpg", 1);	//��ͼ
	img_r = imread ("D:\\Learn_Files\\OpenCV\\StitchDetial\\right.jpg", 1);	//��ͼ

	//ת��Ϊ�Ҷ�ͼ
	cvtColor (img_l, img_grayl, COLOR_BGR2GRAY);
	cvtColor (img_r, img_grayr, COLOR_BGR2GRAY);

	//SIFT���������, ����ؼ���
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create ();

	//���ؼ��㣬�ֱ�洢�� keypoint_l��keypoint_r��
	vector<KeyPoint> keypoint_l, keypoint_r;
	sift->detect (img_grayl, keypoint_l);
	sift->detect (img_grayr, keypoint_r);

	//�����������ӣ��洢��descriptor��
	Mat descriptor_l, descriptor_r;
	sift->compute (img_grayl, keypoint_l, descriptor_l);
	sift->compute (img_grayr, keypoint_r, descriptor_r);

	//���ٽ������ڽ���Ѱ��ƥ���
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> matchePoints;
	vector<DMatch> good_match_points;		//�洢�õ�ƥ���

	vector<Mat> train_desc (1, descriptor_r);

	matcher.add (train_desc);		//���ѵ��������
	//KNNMatch��������K = 2 ������ÿ��ƥ�䷵�������������������������һ��ƥ����ڶ���ƥ��֮��ľ����㹻Сʱ������Ϊ����һ��ƥ�䡣
	matcher.knnMatch (descriptor_l, matchePoints, 2);
	cout << "total match points: " << matchePoints.size () << endl;

	//��ȡ����ƥ���
	for (int i = 0; i < matchePoints.size (); i++)
	{
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
		{
			good_match_points.push_back (matchePoints[i][0]);
		}
	}

	//��ʾ�ؼ���
	drawKeypoints (img_l, keypoint_l, descriptor_l, cv::Scalar::all (-1));
	drawKeypoints (img_r, keypoint_r, descriptor_r, cv::Scalar::all (-1));

	//����������ƥ��
	drawMatches (img_l, keypoint_l, img_r, keypoint_r, good_match_points, LRmatcher, cv::Scalar::all (-1), cv::Scalar (0, 0, 255));

	vector<Point2f> kp_l, kp_r;
	for (int i = 0; i < good_match_points.size (); i++)
	{
		kp_r.push_back (keypoint_r[good_match_points[i].trainIdx].pt);	//��ƥ���������
		kp_l.push_back (keypoint_l[good_match_points[i].queryIdx].pt);	//Ҫƥ���������
	}

	//�õ��任����
	Mat homography = findHomography (kp_r, kp_l, noArray (), RANSAC);
	cout << "�任����:		\n" << homography << endl << endl;

	Mat result;		//���ս��ͼƬ

	//������ͼ����ͼ
	warpPerspective (img_r, result, homography, Size (img_l.cols + img_r.cols, img_l.rows));
	//imshow ("Left Image:", result);

	// ����ͼ���Ƶ�ǰ�沿�֣�Ȼ�����ͼ�Ž�ȥ
	Mat half (result, Rect (0, 0, img_l.cols, img_l.rows));

	//������ͼ����ͼ��ROI����
	img_l.copyTo (half);


	//imshow ("��ͼ-img_l", descriptor_l);
	//imshow ("��ͼ-img_r", descriptor_r);
	imshow ("ƥ����", LRmatcher);
	imshow ("Result", result);

	cv::imwrite ("Result.jpg", result);

	waitKey (0);
}
