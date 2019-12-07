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
	img_l = imread ("D:\\Learn_Files\\OpenCV\\StitchDetial\\left.jpg", 1);	//左图
	img_r = imread ("D:\\Learn_Files\\OpenCV\\StitchDetial\\right.jpg", 1);	//右图

	//转化为灰度图
	cvtColor (img_l, img_grayl, COLOR_BGR2GRAY);
	cvtColor (img_r, img_grayr, COLOR_BGR2GRAY);

	//SIFT特征检测器, 计算关键点
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create ();

	//检测关键点，分别存储到 keypoint_l和keypoint_r中
	vector<KeyPoint> keypoint_l, keypoint_r;
	sift->detect (img_grayl, keypoint_l);
	sift->detect (img_grayr, keypoint_r);

	//计算描述算子，存储在descriptor中
	Mat descriptor_l, descriptor_r;
	sift->compute (img_grayl, keypoint_l, descriptor_l);
	sift->compute (img_grayr, keypoint_r, descriptor_r);

	//快速近似最邻近，寻找匹配点
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> matchePoints;
	vector<DMatch> good_match_points;		//存储好的匹配点

	vector<Mat> train_desc (1, descriptor_r);

	matcher.add (train_desc);		//添加训练描述符
	//KNNMatch，可设置K = 2 ，即对每个匹配返回两个最近邻描述符，仅当第一个匹配与第二个匹配之间的距离足够小时，才认为这是一个匹配。
	matcher.knnMatch (descriptor_l, matchePoints, 2);
	cout << "total match points: " << matchePoints.size () << endl;

	//获取优秀匹配点
	for (int i = 0; i < matchePoints.size (); i++)
	{
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
		{
			good_match_points.push_back (matchePoints[i][0]);
		}
	}

	//显示关键点
	drawKeypoints (img_l, keypoint_l, descriptor_l, cv::Scalar::all (-1));
	drawKeypoints (img_r, keypoint_r, descriptor_r, cv::Scalar::all (-1));

	//特征点连线匹配
	drawMatches (img_l, keypoint_l, img_r, keypoint_r, good_match_points, LRmatcher, cv::Scalar::all (-1), cv::Scalar (0, 0, 255));

	vector<Point2f> kp_l, kp_r;
	for (int i = 0; i < good_match_points.size (); i++)
	{
		kp_r.push_back (keypoint_r[good_match_points[i].trainIdx].pt);	//被匹配的描述子
		kp_l.push_back (keypoint_l[good_match_points[i].queryIdx].pt);	//要匹配的描述子
	}

	//得到变换矩阵
	Mat homography = findHomography (kp_r, kp_l, noArray (), RANSAC);
	cout << "变换矩阵:		\n" << homography << endl << endl;

	Mat result;		//最终结果图片

	//歪曲右图到左图
	warpPerspective (img_r, result, homography, Size (img_l.cols + img_r.cols, img_l.rows));
	//imshow ("Left Image:", result);

	// 把右图复制到前面部分，然后把左图放进去
	Mat half (result, Rect (0, 0, img_l.cols, img_l.rows));

	//复制左图到右图的ROI区域
	img_l.copyTo (half);


	//imshow ("左图-img_l", descriptor_l);
	//imshow ("右图-img_r", descriptor_r);
	imshow ("匹配线", LRmatcher);
	imshow ("Result", result);

	cv::imwrite ("Result.jpg", result);

	waitKey (0);
}
