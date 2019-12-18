#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

#define PosSamNO 3350    //正样本个数
#define NegSamNO 15633   //负样本个数
#define TRAIN false		 //true表示重新训练，false表示读取xml文件中的SVM模型
#define HardExampleNO 37	

class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double* get_alpha_vector ()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	double get_rho ()
	{
		return this->decision_func->rho;
	}
};

int main ()
{
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog (Size (64, 128), Size (16, 16), Size (8, 8), Size (8, 8), 9);
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
	if (TRAIN)
	{
		string ImgName;				//图片名(绝对路径)
		ifstream finPos ("pos.txt");		//正样本图片的文件名列表
		ifstream finNeg ("neg.txt");		//负样本图片的文件名列表
		Mat sampleFeatureMat;		//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
		Mat sampleLabelMat;			//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人
		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num < PosSamNO && getline (finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			Mat src = imread (ImgName);		//读取图片

			vector<float> descriptors;		//HOG描述子向量
			hog.compute (src, descriptors, Size (8, 8));	//计算HOG描述子，检测窗口移动步长(8,8)
			cout << "描述子维数：" << descriptors.size () << endl;

			if (num == 0)
			{
				DescriptorDim = descriptors.size ();		//	HOG描述子的维数
				//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat

				sampleFeatureMat = Mat::zeros (PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);

				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros (PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++)
				sampleFeatureMat.at<float> (num, i) = descriptors[i];	//第num个样本的特征向量中的第i个元素

			sampleLabelMat.at<float> (num, 0) = 1;		//正样本类别为1，有人
		}

		
		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num < NegSamNO && getline (finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			Mat src = imread (ImgName);//读取图片

			vector<float> descriptors;//HOG描述子向量
			hog.compute (src, descriptors, Size (8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++)
				//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleFeatureMat.at<float> (num + PosSamNO, i) = descriptors[i];

			sampleLabelMat.at<float> (num + PosSamNO, 0) = -1;//负样本类别为-1，无人
		}
		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			ifstream finHardExample ("hard.txt");//HardExample负样本的文件名列表
			//依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num < HardExampleNO && getline (finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				//ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread (ImgName);//读取图像

				vector<float> descriptors;//HOG描述子向量
				hog.compute (src, descriptors, Size (8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for (int i = 0; i < DescriptorDim; i++)
					sampleFeatureMat.at<float> (num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float> (num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}
		//训练SVM分类器，迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

		//SVM参数：SVM类型为C_SVC；线性核函数；惩罚因子C=0.01
		CvSVMParams param (CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

		cout << "开始训练SVM分类器" << endl;
		svm.train (sampleFeatureMat, sampleLabelMat, Mat (), Mat (), param);//训练分类器
		cout << "训练完成" << endl;
		svm.save ("SVM_HOG_train.xml");//将训练好的SVM模型保存为xml文件
	}
	else 
	{
		svm.load ("SVM_HOG.xml");//从XML文件读取训练好的SVM模型
	}

	/*线性SVM训练完成后得到的XML文件里面，有数组support vector，数组alpha，浮点数rho；
	* 将alpha矩阵同support vector相乘得到一个列向量。再该列向量的最后添加一个元素rho；
	* 从而生成一个分类器，利用该分类器就可以进行行人检测了。
	*/
	DescriptorDim = svm.get_var_count ();//HOG描述子的维数

	int supportVectorNum = svm.get_support_vector_count ();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros (1, supportVectorNum, CV_32FC1);	//alpha向量, 长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros (supportVectorNum, DescriptorDim, CV_32FC1);	//支持向量矩阵
	Mat resultMat = Mat::zeros (1, DescriptorDim, CV_32FC1);	//alpha向量乘以支持向量矩阵的结果

	//将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float* pSVData = svm.get_support_vector (i);	//返回第i个支持向量的数据指针
		for (int j = 0; j < DescriptorDim; j++)
		{
			supportVectorMat.at<float> (i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double* pAlphaData = svm.get_alpha_vector ();
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float> (0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的可用检测子
	vector<float> myDetector;

	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back (resultMat.at<float> (0, i));
	}

	//添加偏移量rho，得到检测子
	myDetector.push_back (svm.get_rho ());
	cout << "检测子维数：" << myDetector.size () << endl;

	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector (myDetector);

	//保存检测子参数到文件
	ofstream fout ("HOGDetectorForOpenCV.txt");
	for (unsigned int i = 0; i < myDetector.size (); i++)
	{
		fout << myDetector[i] << endl;
	}


	/*    读入图像进行HOG行人检测   */
	Mat src = imread ("pic_280.jpg");

	vector<Rect> found, found_filtered;		//矩阵框数组

	cout << "进行多尺度HOG人体检测" << endl;
	/*src为输入待检测的图片；
	 *found为检测到目标区域列表；
	 *参数3为程序内部计算为行人目标的阈值，也就是检测到的特征到SVM分类超平面的距离;
	 *参数4为滑动窗口每次移动的距离。它必须是块移动的整数倍；
	 *参数5为图像扩充的大小；
	 *参数6为比例系数，即测试图片每次尺寸缩放增加的比例；
	* 参数7为组阈值，即校正系数，当一个目标被多个窗口检测出来时，该参数此时就起了调节作用，为0时表示不起调节作用。
	*/
	myHOG.detectMultiScale (src, found, 0, Size (8, 8), Size (32, 32), 1.05, 2);
	cout << "找到的矩形框个数：" << found.size () << endl;

	//最大的矩形框放入found_filtered
	for (unsigned int i = 0; i < found.size (); i++)
	{
		Rect r = found[i];
		unsigned int j = 0;
		for (; j < found.size (); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size ())
			found_filtered.push_back (r);
	}

	//画矩形框
	for (unsigned int i = 0; i < found_filtered.size (); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound (r.width * 0.1);
		r.width = cvRound (r.width * 0.8);
		r.y += cvRound (r.height * 0.07);
		r.height = cvRound (r.height * 0.8);
		rectangle (src, r.tl (), r.br (), Scalar (0, 255, 0), 3);
	}
	imwrite ("Img.jpg", src);
	namedWindow ("src", 0);
	imshow ("src", src);
	waitKey (0);
}
