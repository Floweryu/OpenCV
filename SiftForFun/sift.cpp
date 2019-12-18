#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

int main ()
{
	Mat image1 = imread ("test1.jpg");
	Mat image2 = imread ("test2.jpg");
	if (!image1.data || !image2.data) {
		cout << "Reading images errror !!!" << endl;
	}

	int numFeatures = 100;		// ������ĸ���
	int minHessian = 40;
	cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create (numFeatures);
	//������
	std::vector<cv::KeyPoint> keyPointL, keyPointR;
	//������ȡ������
	sift->detect (image1, keyPointL);
	sift->detect (image2, keyPointR);
	//��������
	cv::Mat keyPointImageL;
	cv::Mat keyPointImageR;
	drawKeypoints (image1, keyPointL, keyPointImageL, cv::Scalar::all (-1), cv::DrawMatchesFlags::DEFAULT);
	drawKeypoints (image2, keyPointR, keyPointImageR, cv::Scalar::all (-1), cv::DrawMatchesFlags::DEFAULT);
	//��ʾ����
	cv::namedWindow ("KeyPoints of imageL");
	cv::namedWindow ("KeyPoints of imageR");

	//��ʾ������
	cv::imshow ("KeyPoints of imageL", keyPointImageL);
	cv::imshow ("KeyPoints of imageR", keyPointImageR);

	//������ƥ��
	cv::Mat despL, despR;
	//��ȡ�����㲢��������������
	sift->detectAndCompute (image1, cv::Mat (), keyPointL, despL);
	sift->detectAndCompute (image2, cv::Mat (), keyPointR, despR);
	std::vector<cv::DMatch> matches;

	//�������flannBased���� ��ô despͨ��orb�ĵ������Ͳ�ͬ��Ҫ��ת������
	if (despL.type () != CV_32F || despR.type () != CV_32F)
	{
		despL.convertTo (despL, CV_32F);
		despR.convertTo (despR, CV_32F);
	}

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create ("FlannBased");
	matcher->match (despL, despR, matches);

	//�����������������ֵ 
	double maxDist = 0;
	for (int i = 0; i < despL.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist)
			maxDist = dist;
	}

	//��ѡ�õ�ƥ���
	std::vector< cv::DMatch > good_matches;
	for (int i = 0; i < despL.rows; i++)
	{
		if (matches[i].distance < 0.5 * maxDist)
		{
			good_matches.push_back (matches[i]);
		}
	}

	cv::Mat imageOutput;
	cv::drawMatches (image1, keyPointL, image2, keyPointR, good_matches, imageOutput);

	cv::namedWindow ("picture of matching");
	cv::imshow ("picture of matching", imageOutput);
	cv::waitKey (0);
	return 0;
}