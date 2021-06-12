#include "pch.h"
#include "Model.h"

using namespace std;
using namespace cv;

inline void JPEG(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm, string strNetPath);

/******************************************�㷨��*/
// �㷨��Ƭ��
class Algorithm_RDDLN_JPEGlv1_Photo :public MAlgorithm
{
public:
	Algorithm_RDDLN_JPEGlv1_Photo();

	BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm);
private:
	/* data */
};

// �㷨��Ƭ��
class Algorithm_RDDLN_JPEGlv2_Photo :public MAlgorithm
{
public:
	Algorithm_RDDLN_JPEGlv2_Photo();

	BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm);
private:
	/* data */
};

// �㷨������
class Algorithm_RDDLN_JPEGlv1_Anime :public MAlgorithm
{
public:
	Algorithm_RDDLN_JPEGlv1_Anime();

	BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm);
private:
	/* data */
};

class Algorithm_RDDLN_JPEGlv2_Anime :public MAlgorithm
{
public:
	Algorithm_RDDLN_JPEGlv2_Anime();

	BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm);
private:
	/* data */
};


// ��ʼ�������㷨����
Algorithm_RDDLN_JPEGlv1_Photo::Algorithm_RDDLN_JPEGlv1_Photo()
{
	strAlgorithmName = "�޸�JPEGʧ��(LV1)-��Ƭ";
}


BOOL Algorithm_RDDLN_JPEGlv1_Photo::Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm)
{
	// ��ʼ��
	string strNetPath = R"(.\Model\RDDLN\RDDLN_JPEG_lv1_photo.onnx)";
	JPEG(ptrBridge, s_AlgorithmParm, strNetPath);

	return 0;
}

// ��ʼ�������㷨����
Algorithm_RDDLN_JPEGlv2_Photo::Algorithm_RDDLN_JPEGlv2_Photo()
{
	strAlgorithmName = "�޸�JPEGʧ��(LV2)-��Ƭ";
}


BOOL Algorithm_RDDLN_JPEGlv2_Photo::Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm)
{
	// ��ʼ��
	string strNetPath = R"(.\Model\RDDLN\RDDLN_JPEG_lv2_photo.onnx)";
	JPEG(ptrBridge, s_AlgorithmParm, strNetPath);

	return 0;
}

// ��ʼ�������㷨����
Algorithm_RDDLN_JPEGlv1_Anime::Algorithm_RDDLN_JPEGlv1_Anime()
{
	strAlgorithmName = "�޸�JPEGʧ��(LV1)-����";
}


BOOL Algorithm_RDDLN_JPEGlv1_Anime::Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm)
{
	// ��ʼ��
	string strNetPath = R"(.\Model\RDDLN\RDDLN_JPEG_lv1_anime.onnx)";
	JPEG(ptrBridge, s_AlgorithmParm, strNetPath);

	return 0;
}

// ��ʼ�������㷨����
Algorithm_RDDLN_JPEGlv2_Anime::Algorithm_RDDLN_JPEGlv2_Anime()
{
	strAlgorithmName = "�޸�JPEGʧ��(LV2)-����";
}


BOOL Algorithm_RDDLN_JPEGlv2_Anime::Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm)
{	
	// ��ʼ��
	string strNetPath = R"(.\Model\RDDLN\RDDLN_JPEG_lv2_anime.onnx)";
	JPEG(ptrBridge, s_AlgorithmParm, strNetPath);

	return 0;
}



/*******************************************RDDLN�޸�JPEG*/
RDDLN_JPEG::RDDLN_JPEG()
{
	strModelName = "RDDLN-JPEG";
	vctAlgorithms.push_back(make_shared<Algorithm_RDDLN_JPEGlv1_Photo>());
	vctAlgorithms.push_back(make_shared<Algorithm_RDDLN_JPEGlv2_Photo>());
	vctAlgorithms.push_back(make_shared<Algorithm_RDDLN_JPEGlv1_Anime>());
	vctAlgorithms.push_back(make_shared<Algorithm_RDDLN_JPEGlv2_Anime>());
}

RDDLN_JPEG::~RDDLN_JPEG()
{
}


inline uchar float2uchar(float dInPix)
{
	float dPix = dInPix * 255.f + 0.5f;
	dPix = dPix < 0.f ? 0.f : dPix;
	dPix = dPix > 255.f ? 255.f : dPix;

	uchar Pix = uchar(dPix);

	return Pix;
}

inline void JPEG(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm, string strNetPath)
{
	constexpr uint32_t iSize_LR = 64;

	constexpr uint32_t iSize_SR = 60;

	constexpr uint32_t iStride = 48;

	constexpr uint32_t iScale = 1;

	constexpr uint32_t iPadSize = (iSize_LR * iScale - iSize_SR) / (iScale * 2);

	constexpr uint32_t iStride_SR = iStride * iScale;

	constexpr uint32_t iEdgeSize = (iSize_SR - iStride_SR) / 2;

	uint32_t iBatchSize = max(s_AlgorithmParm.iBatchSize / 2, UINT(1));

	try
	{
		// ����ģ��
		dnn::Net net = dnn::readNetFromONNX(strNetPath);

		if (s_AlgorithmParm.bCUDA)
		{
			// ����������ʹ��CUDA���ʱʹ��
			net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
		}

		// ��ȡͼƬ
		Mat imgOrigin = imread(s_AlgorithmParm.strInputPath);

		// ��ȡͼ��ԭʼ�ߴ�
		uint32_t iOriginWidth = imgOrigin.cols;
		uint32_t iOriginHeight = imgOrigin.rows;

		// ת��ΪYUV����
		cvtColor(imgOrigin, imgOrigin, COLOR_BGR2YUV);

		// ת��Ϊ������
		imgOrigin.convertTo(imgOrigin, CV_32F);

		// ��һ��
		// normalize(imgOrigin, imgOrigin, 1.0, 0.0, NORM_MINMAX);

		// ͼ���С(С�ڵ�λ������)�����ٴ���
		bool bCutFlag = false;// ���ͼ���С,�������һ�βü�����
		if (iOriginWidth < (iSize_LR - iPadSize * 2))// ��Ȳ���
		{
			// ����ұߵ�(iSize_LR - iPadSize * 2)��С
			copyMakeBorder(imgOrigin, imgOrigin, 0, 0, 0, (iSize_LR - iPadSize * 2) - iOriginWidth, BORDER_REPLICATE);
			bCutFlag = true;
		}
		if (iOriginHeight < (iSize_LR - iPadSize * 2))// �߶Ȳ���
		{
			// ����±ߵ�(iSize_LR - iPadSize * 2)��С
			copyMakeBorder(imgOrigin, imgOrigin, 0, (iSize_LR - iPadSize * 2) - iOriginHeight, 0, 0, BORDER_REPLICATE);
			bCutFlag = true;
		}

		// �����Ե
		copyMakeBorder(imgOrigin, imgOrigin, iPadSize, iPadSize, iPadSize, iPadSize, BORDER_REFLECT_101);

		// ��ȡͼ��ߴ�
		uint32_t iWidth = imgOrigin.cols;
		uint32_t iHeight = imgOrigin.rows;

		// ����ֿ��Ŀ�߿���
		uint32_t iWidthStep = (iWidth + iStride - iSize_LR) / iStride;
		uint32_t iHeightStep = (iHeight + iStride - iSize_LR) / iStride;

		// �����ؽ��󳬷�ͼ��ߴ�
		uint32_t iWidth_SR = (iWidth - (iPadSize * 2)) * iScale;
		uint32_t iHeight_SR = (iHeight - (iPadSize * 2)) * iScale;

		// ���������ͼ����
		vector<Mat> vImgSub;

		// �����������ͼ�����
		queue<Mat> qImgSub_SR;

		// ���ֺ��ͨ������
		vector<Mat> vImgChannels_SR;

		// �����ؽ���ͼ��
		Mat imgSR(iHeight_SR, iWidth_SR, CV_8UC3);

		for (uint32_t col = 0; col <= iWidthStep; col++)
		{
			for (uint32_t row = 0; row <= iHeightStep; row++)
			{
				// �����и�����
				// ��ʼ�����
				uint32_t x = col * iStride;
				uint32_t y = row * iStride;

				// ��������һ��
				if (col == iWidthStep)
				{
					x = iWidth - iSize_LR;
				}
				// ��������һ��
				if (row == iHeightStep)
				{
					y = iHeight - iSize_LR;
				}

				// ��ȡ����
				Rect rectSub = Rect(x, y, iSize_LR, iSize_LR);

				Mat imgSub;
				imgOrigin(rectSub).copyTo(imgSub);

				// ��ȡ��ͼ��
				vImgSub.push_back(imgSub);

				// ����(ռ10%)
				double_t Progress = (double_t(col * iHeightStep + row)) / // ��ǰͨ���Ѵ������е�������
					double_t(iWidthStep * iHeightStep) * 0.1;// �ܹ���������
				ptrBridge->SetProgress(Progress);

				// ���FlagΪ��������
				if (ptrBridge->StopFlag())
					return;
			}
		}

		// ��batch size��ͼ���г��ֱ���
		size_t iBatchStep = vImgSub.size() / iBatchSize;
		for (size_t step = 0; step <= iBatchStep; step++)
		{
			// �����α�����ͼ���������ʼλ�úͽ���λ��
			auto start = vImgSub.begin() + (step * iBatchSize);
			auto end = start;

			// ��������һ��batch
			if (step == iBatchStep)
			{
				// ������ͼ������Ľ���λ��
				end = vImgSub.end();
			}
			else
			{
				// ������ͼ������Ľ���λ��
				end = start + iBatchSize;
			}

			// ���û��ʣ��
			if (start == end)
			{
				// ֱ������
				break;
			}

			// ��ȡһ����ͼ��
			vector<Mat> vImgBatch(start, end);

			// ����size
			Size size = Size(iSize_LR, iSize_LR);

			// ת��Ϊblob
			Mat blob = dnn::blobFromImages(vImgBatch, 1.0 / 255.f, size);

			// ������������
			net.setInput(blob);

			// ���򴫲�
			Mat SR = net.forward();

			// ת��Ϊͼ��
			vector<Mat> vImgBatch_SR;
			dnn::imagesFromBlob(SR, vImgBatch_SR);

			// �������
			for (auto imgSR : vImgBatch_SR)
			{
				qImgSub_SR.push(imgSR);
			}

			// ���FlagΪ��������
			if (ptrBridge->StopFlag())
				return;

			// ����(ռ60%)
			double_t Progress = 0.1 + ((double_t)step / (double_t)iBatchStep) * 0.6;
			ptrBridge->SetProgress(Progress);
		}

		// �ؽ�SRͼ��
		for (uint32_t col = 0; col <= iWidthStep; col++)
		{
			for (uint32_t row = 0; row <= iHeightStep; row++)
			{
				// ȡ�����е�һ��ֵ
				Mat Sub_SR = qImgSub_SR.front();
				qImgSub_SR.pop();

				// �ؽ�������SRͼ��λ��
				uint32_t x = col * iStride_SR;
				uint32_t y = row * iStride_SR;
				// ����ؽ�λ���ڱ�Ե
				if (col == iWidthStep)
				{
					x = iWidth_SR - iSize_SR;
				}
				if (row == iHeightStep)
				{
					y = iHeight_SR - iSize_SR;
				}

				// ��Ҫƽ����Եλ��
				uint32_t w_start = iEdgeSize;
				uint32_t h_start = iEdgeSize;
				// ���ƴ�ӵ�λ������ʼ��Ե
				if (col == 0)
				{
					w_start = 0;
				}
				if (row == 0)
				{
					h_start = 0;
				}

				// ƴ��ͼ��
				for (uint32_t w = w_start; w < iSize_SR; w++)
				{
					for (uint32_t h = h_start; h < iSize_SR; h++)
					{
						for (uint32_t c = 0; c < imgSR.channels(); c++)
						{
							imgSR.at<Vec3b>(y + h, x + w)[c] = float2uchar(Sub_SR.at<Vec3f>(h, w)[c]);
						}
					}
				}

				// ƽ����ֱ��Ե
				for (uint32_t w = 0; w < iEdgeSize; w++)
				{
					for (uint32_t h = 0; h < iSize_SR; h++)
					{
						for (uint32_t c = 0; c < imgSR.channels(); c++)
						{
							// ��ǰ����ֵ
							float pix = float(imgSR.at<Vec3b>(y + h, x + w)[c]) / 255.f;
							// ƽ��ϵ��
							float alpha = float(w) / float(iEdgeSize);
							imgSR.at<Vec3b>(y + h, x + w)[c] = float2uchar(Sub_SR.at<Vec3f>(h, w)[c] * alpha + pix * (1 - alpha));
						}
					}
				}

				// ƽ��ˮƽ��Ե
				for (uint32_t w = 0; w < iSize_SR; w++)
				{
					for (uint32_t h = 0; h < iEdgeSize; h++)
					{
						for (uint32_t c = 0; c < imgSR.channels(); c++)
						{
							// ��ǰ����ֵ
							float pix = float(imgSR.at<Vec3b>(y + h, x + w)[c]) / 255.f;
							// ƽ��ϵ��
							float alpha = float(h) / float(iEdgeSize);
							imgSR.at<Vec3b>(y + h, x + w)[c] = float2uchar(Sub_SR.at<Vec3f>(h, w)[c] * alpha + pix * (1 - alpha));
						}
					}
				}

				// ����(ռ30%)
				double_t Progress = 0.7 + (double_t(col * iHeightStep + row)) / // ��ǰͨ���Ѵ������е�������
					double_t(iWidthStep * iHeightStep) * 0.3;// �ܹ���������
				ptrBridge->SetProgress(Progress);

				// ���FlagΪ��������
				if (ptrBridge->StopFlag())
					return;
			}
		}

		// YUVת��ΪBGR(RGB)
		cvtColor(imgSR, imgSR, COLOR_YUV2BGR);

		// ��ȡ����
		Rect rectSub = Rect(0, 0, iOriginWidth, iOriginHeight);

		if (bCutFlag)
		{
			// �ü�
			imgSR = imgSR(rectSub);
		}

		// ����ͼ��
		imwrite(s_AlgorithmParm.strOutputPath, imgSR);

		// ����ź�
		ptrBridge->Accomplish();
	}
	catch (cv::Exception& e)
	{
		// �쳣
		ptrBridge->operator<<(Log(FAILED_MODEL_LOAD, e.what()));

	}
}