#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

#define PosSamNO 3350    //����������
#define NegSamNO 15633   //����������
#define TRAIN false		 //true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define HardExampleNO 37	

class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double* get_alpha_vector ()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	double get_rho ()
	{
		return this->decision_func->rho;
	}
};

int main ()
{
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog (Size (64, 128), Size (16, 16), Size (8, 8), Size (8, 8), 9);
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������
	if (TRAIN)
	{
		string ImgName;				//ͼƬ��(����·��)
		ifstream finPos ("pos.txt");		//������ͼƬ���ļ����б�
		ifstream finNeg ("neg.txt");		//������ͼƬ���ļ����б�
		Mat sampleFeatureMat;		//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
		Mat sampleLabelMat;			//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����
		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num < PosSamNO && getline (finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			Mat src = imread (ImgName);		//��ȡͼƬ

			vector<float> descriptors;		//HOG����������
			hog.compute (src, descriptors, Size (8, 8));	//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			cout << "������ά����" << descriptors.size () << endl;

			if (num == 0)
			{
				DescriptorDim = descriptors.size ();		//	HOG�����ӵ�ά��
				//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat

				sampleFeatureMat = Mat::zeros (PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);

				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
				sampleLabelMat = Mat::zeros (PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++)
				sampleFeatureMat.at<float> (num, i) = descriptors[i];	//��num�����������������еĵ�i��Ԫ��

			sampleLabelMat.at<float> (num, 0) = 1;		//���������Ϊ1������
		}

		
		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num < NegSamNO && getline (finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			Mat src = imread (ImgName);//��ȡͼƬ

			vector<float> descriptors;//HOG����������
			hog.compute (src, descriptors, Size (8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++)
				//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleFeatureMat.at<float> (num + PosSamNO, i) = descriptors[i];

			sampleLabelMat.at<float> (num + PosSamNO, 0) = -1;//���������Ϊ-1������
		}
		//����HardExample������
		if (HardExampleNO > 0)
		{
			ifstream finHardExample ("hard.txt");//HardExample���������ļ����б�
			//���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num < HardExampleNO && getline (finHardExample, ImgName); num++)
			{
				cout << "����" << ImgName << endl;
				//ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����
				Mat src = imread (ImgName);//��ȡͼ��

				vector<float> descriptors;//HOG����������
				hog.compute (src, descriptors, Size (8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

				//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
				for (int i = 0; i < DescriptorDim; i++)
					sampleFeatureMat.at<float> (num + PosSamNO + NegSamNO, i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
				sampleLabelMat.at<float> (num + PosSamNO + NegSamNO, 0) = -1;//���������Ϊ-1������
			}
		}
		//ѵ��SVM��������������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

		//SVM������SVM����ΪC_SVC�����Ժ˺������ͷ�����C=0.01
		CvSVMParams param (CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

		cout << "��ʼѵ��SVM������" << endl;
		svm.train (sampleFeatureMat, sampleLabelMat, Mat (), Mat (), param);//ѵ��������
		cout << "ѵ�����" << endl;
		svm.save ("SVM_HOG_train.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	}
	else 
	{
		svm.load ("SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	/*����SVMѵ����ɺ�õ���XML�ļ����棬������support vector������alpha��������rho��
	* ��alpha����ͬsupport vector��˵õ�һ�����������ٸ���������������һ��Ԫ��rho��
	* �Ӷ�����һ�������������ø÷������Ϳ��Խ������˼���ˡ�
	*/
	DescriptorDim = svm.get_var_count ();//HOG�����ӵ�ά��

	int supportVectorNum = svm.get_support_vector_count ();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros (1, supportVectorNum, CV_32FC1);	//alpha����, ���ȵ���֧����������
	Mat supportVectorMat = Mat::zeros (supportVectorNum, DescriptorDim, CV_32FC1);	//֧����������
	Mat resultMat = Mat::zeros (1, DescriptorDim, CV_32FC1);	//alpha��������֧����������Ľ��

	//��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float* pSVData = svm.get_support_vector (i);	//���ص�i��֧������������ָ��
		for (int j = 0; j < DescriptorDim; j++)
		{
			supportVectorMat.at<float> (i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double* pAlphaData = svm.get_alpha_vector ();
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float> (0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յĿ��ü����
	vector<float> myDetector;

	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back (resultMat.at<float> (0, i));
	}

	//���ƫ����rho���õ������
	myDetector.push_back (svm.get_rho ());
	cout << "�����ά����" << myDetector.size () << endl;

	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG;
	myHOG.setSVMDetector (myDetector);

	//�������Ӳ������ļ�
	ofstream fout ("HOGDetectorForOpenCV.txt");
	for (unsigned int i = 0; i < myDetector.size (); i++)
	{
		fout << myDetector[i] << endl;
	}


	/*    ����ͼ�����HOG���˼��   */
	Mat src = imread ("pic_280.jpg");

	vector<Rect> found, found_filtered;		//���������

	cout << "���ж�߶�HOG������" << endl;
	/*srcΪ���������ͼƬ��
	 *foundΪ��⵽Ŀ�������б�
	 *����3Ϊ�����ڲ�����Ϊ����Ŀ�����ֵ��Ҳ���Ǽ�⵽��������SVM���೬ƽ��ľ���;
	 *����4Ϊ��������ÿ���ƶ��ľ��롣�������ǿ��ƶ�����������
	 *����5Ϊͼ������Ĵ�С��
	 *����6Ϊ����ϵ����������ͼƬÿ�γߴ��������ӵı�����
	* ����7Ϊ����ֵ����У��ϵ������һ��Ŀ�걻������ڼ�����ʱ���ò�����ʱ�����˵������ã�Ϊ0ʱ��ʾ����������á�
	*/
	myHOG.detectMultiScale (src, found, 0, Size (8, 8), Size (32, 32), 1.05, 2);
	cout << "�ҵ��ľ��ο������" << found.size () << endl;

	//���ľ��ο����found_filtered
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

	//�����ο�
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
