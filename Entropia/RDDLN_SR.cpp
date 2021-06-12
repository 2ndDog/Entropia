#include "pch.h"
#include "Model.h"

using namespace std;
using namespace cv;

inline void SRx2(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm, string strNetPath);

/******************************************算法类*/
// 算法照片类
class Algorithm_RDDLN_SRx2_Photo :public MAlgorithm
{
public:
	Algorithm_RDDLN_SRx2_Photo();

	// 执行算法
	BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm);
};

// 算法动漫类
class Algorithm_RDDLN_SRx2_Anime :public MAlgorithm
{
public:
	Algorithm_RDDLN_SRx2_Anime();

	BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm);
};


// 初始化设置算法名称
Algorithm_RDDLN_SRx2_Photo::Algorithm_RDDLN_SRx2_Photo()
{
	strAlgorithmName = "超分辨率x2-照片";
}

BOOL Algorithm_RDDLN_SRx2_Photo::Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm)
{
	// 初始化
	string strNetPath = R"(.\Model\RDDLN\RDDLNx2_photo.onnx)";
	SRx2(ptrBridge, s_AlgorithmParm, strNetPath);

	return 0;
}


// 初始化设置算法名称
Algorithm_RDDLN_SRx2_Anime::Algorithm_RDDLN_SRx2_Anime()
{
	strAlgorithmName = "超分辨率x2-动漫";
}

BOOL Algorithm_RDDLN_SRx2_Anime::Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm)
{
	// 初始化
	string strNetPath = R"(.\Model\RDDLN\RDDLNx2_anime.onnx)";
	SRx2(ptrBridge, s_AlgorithmParm, strNetPath);

	return 0;
}


/*******************************************RDDLN超分辨率*/
RDDLN_SR::RDDLN_SR()
{
	strModelName = "RDDLN-超分辨率";
	vctAlgorithms.push_back(make_shared<Algorithm_RDDLN_SRx2_Photo>());
	vctAlgorithms.push_back(make_shared<Algorithm_RDDLN_SRx2_Anime>());
}

RDDLN_SR::~RDDLN_SR()
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

inline void SRx2(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm, string strNetPath)
{
	constexpr uint32_t iSize_LR = 64;

	constexpr uint32_t iSize_SR = 120;

	constexpr uint32_t iStride = 48;

	constexpr uint32_t iScale = 2;

	constexpr uint32_t iPadSize = (iSize_LR * iScale - iSize_SR) / (iScale * 2);

	constexpr uint32_t iStride_SR = iStride * iScale;

	constexpr uint32_t iEdgeSize = (iSize_SR - iStride_SR) / 2;

	uint32_t iBatchSize = max(s_AlgorithmParm.iBatchSize / 2, UINT(1));

	try
	{
		// 加载模型
		dnn::Net net = dnn::readNetFromONNX(strNetPath);

		// 是否使用CUDA
		if (s_AlgorithmParm.bCUDA)
		{
			// 下面两行在使用CUDA检测时使用
			net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
		}

		// 读取图片
		Mat imgOrigin = imread(s_AlgorithmParm.strInputPath);

		// 获取图像原始尺寸
		uint32_t iOriginWidth = imgOrigin.cols;
		uint32_t iOriginHeight = imgOrigin.rows;

		// 转化为YUV处理
		cvtColor(imgOrigin, imgOrigin, COLOR_BGR2YUV);

		// 转化为浮点数
		imgOrigin.convertTo(imgOrigin, CV_32F);

		// 归一化
		// normalize(imgOrigin, imgOrigin, 1.0, 0.0, NORM_MINMAX);

		// 图像过小(小于单位块像素)填充后再处理
		bool bCutFlag = false;// 如果图像过小,则最后做一次裁剪操作
		if (iOriginWidth < (iSize_LR - iPadSize * 2))// 宽度不足
		{
			// 填充右边到(iSize_LR - iPadSize * 2)大小
			copyMakeBorder(imgOrigin, imgOrigin, 0, 0, 0, (iSize_LR - iPadSize * 2) - iOriginWidth, BORDER_REPLICATE);
			bCutFlag = true;
		}
		if (iOriginHeight < (iSize_LR - iPadSize * 2))// 高度不足
		{
			// 填充下边到(iSize_LR - iPadSize * 2)大小
			copyMakeBorder(imgOrigin, imgOrigin, 0, (iSize_LR - iPadSize * 2) - iOriginWidth, 0, 0, BORDER_REPLICATE);
			bCutFlag = true;
		}

		// 扩充边缘
		copyMakeBorder(imgOrigin, imgOrigin, iPadSize, iPadSize, iPadSize, iPadSize, BORDER_REFLECT_101);

		// 通道分离
		vector<Mat> vImgChannels;
		split(imgOrigin, vImgChannels);

		// 获取图像尺寸
		uint32_t iWidth = imgOrigin.cols;
		uint32_t iHeight = imgOrigin.rows;

		// 计算分开的宽高块数
		uint32_t iWidthStep = (iWidth + iStride - iSize_LR) / iStride;
		uint32_t iHeightStep = (iHeight + iStride - iSize_LR) / iStride;

		// 计算重建后超分图像尺寸
		uint32_t iWidth_SR = (iWidth - (iPadSize * 2)) * iScale;
		uint32_t iHeight_SR = (iHeight - (iPadSize * 2)) * iScale;

		// 待处理的子图像组
		vector<Mat> vImgSub;

		// 超分输出的子图像队列
		queue<Mat> qImgSub_SR;

		// 超分后的通道数组
		vector<Mat> vImgChannels_SR;

		// 最终重建的超分辨率图像
		Mat imgSR;

		// 截取子图像
		for (uint32_t c = 0; c < vImgChannels.size(); c++)
		{
			for (uint32_t col = 0; col <= iWidthStep; col++)
			{
				for (uint32_t row = 0; row <= iHeightStep; row++)
				{
					// 计算切割坐标
					// 起始坐标点
					uint32_t x = col * iStride;
					uint32_t y = row * iStride;

					// 如果是最后一列
					if (col == iWidthStep)
					{
						x = iWidth - iSize_LR;
					}
					// 如果是最后一行
					if (row == iHeightStep)
					{
						y = iHeight - iSize_LR;
					}

					// 截取矩形
					Rect rectSub = Rect(x, y, iSize_LR, iSize_LR);

					Mat imgSub;
					vImgChannels.at(c)(rectSub).copyTo(imgSub);

					// 截取子图像
					vImgSub.push_back(imgSub);

					// 进度(占10%)
					double_t Progress = ((double_t(c * iWidthStep * iHeightStep) + // 已经处理多少通道的像素数
						double_t(col * iHeightStep + row)) / // 当前通道已处理行列的像素数
						double_t(vImgChannels.size() * iWidthStep * iHeightStep)) * 0.1;// 总共像素数量
					ptrBridge->SetProgress(Progress);

					// 如果Flag为结束符号
					if (ptrBridge->StopFlag())
						return;
				}
			}
		}

		// 按batch size张图进行超分辨率
		size_t iBatchStep = vImgSub.size() / iBatchSize;
		for (size_t step = 0; step <= iBatchStep; step++)
		{
			// 按批次遍历子图像数组的起始位置和结束位置
			auto start = vImgSub.begin() + (step * iBatchSize);
			auto end = start;

			// 如果是最后一个batch
			if (step == iBatchStep)
			{
				// 遍历子图像数组的结束位置
				end = vImgSub.end();
			}
			else
			{
				// 遍历子图像数组的结束位置
				end = start + iBatchSize;
			}

			// 如果没有剩下
			if (start == end)
			{
				// 直接跳出
				break;
			}

			// 截取一批子图像
			vector<Mat> vImgBatch(start, end);

			// 设置size
			Size size = Size(iSize_LR, iSize_LR);

			// 转化为blob
			Mat blob = dnn::blobFromImages(vImgBatch, 1.0 / 255.f, size);

			// 设置网络输入
			net.setInput(blob);

			// 正向传播
			Mat SR = net.forward();

			// 转化为图像
			vector<Mat> vImgBatch_SR;
			dnn::imagesFromBlob(SR, vImgBatch_SR);

			// 加入队列
			for (auto imgSR : vImgBatch_SR)
			{
				qImgSub_SR.push(imgSR);
			}

			// 如果Flag为结束符号
			if (ptrBridge->StopFlag())
				return;

			// 进度(占60%)
			double_t Progress = 0.1 + ((double_t)step / (double_t)iBatchStep) * 0.6;
			ptrBridge->SetProgress(Progress);
		}

		// 重建SR图像
		for (uint32_t c = 0; c < vImgChannels.size(); c++)
		{
			// 开辟一个单通道空图像
			Mat imgC_SR(iHeight_SR, iWidth_SR, CV_8UC1);

			for (uint32_t col = 0; col <= iWidthStep; col++)
			{
				for (uint32_t row = 0; row <= iHeightStep; row++)
				{
					// 取出队列第一个值
					Mat Sub_SR = qImgSub_SR.front();
					qImgSub_SR.pop();

					// 重建像素在SR图像位置
					uint32_t x = col * iStride_SR;
					uint32_t y = row * iStride_SR;
					// 如果重建位置在边缘
					if (col == iWidthStep)
					{
						x = iWidth_SR - iSize_SR;
					}
					if (row == iHeightStep)
					{
						y = iHeight_SR - iSize_SR;
					}

					// 需要平滑边缘位置
					uint32_t w_start = iEdgeSize;
					uint32_t h_start = iEdgeSize;
					// 如果拼接的位置在起始边缘
					if (col == 0)
					{
						w_start = 0;
					}
					if (row == 0)
					{
						h_start = 0;
					}

					// 拼接图像
					for (uint32_t w = w_start; w < iSize_SR; w++)
					{
						for (uint32_t h = h_start; h < iSize_SR; h++)
						{
							imgC_SR.at<uchar>(y + h, x + w) = float2uchar(Sub_SR.at<float>(h, w));
						}
					}

					// 平滑垂直边缘
					for (uint32_t w = 0; w < iEdgeSize; w++)
					{
						for (uint32_t h = 0; h < iSize_SR; h++)
						{
							// 当前像素值
							float pix = float(imgC_SR.at<uchar>(y + h, x + w)) / 255.f;
							// 平滑系数
							float alpha = float(w) / float(iEdgeSize);
							imgC_SR.at<uchar>(y + h, x + w) = float2uchar(Sub_SR.at<float>(h, w) * alpha + pix * (1 - alpha));
						}
					}

					// 平滑水平边缘
					for (uint32_t w = 0; w < iSize_SR; w++)
					{
						for (uint32_t h = 0; h < iEdgeSize; h++)
						{
							// 当前像素值
							float pix = float(imgC_SR.at<uchar>(y + h, x + w)) / 255.f;
							// 平滑系数
							float alpha = float(h) / float(iEdgeSize);
							imgC_SR.at<uchar>(y + h, x + w) = float2uchar(Sub_SR.at<float>(h, w) * alpha + pix * (1 - alpha));
						}
					}

					// 进度(占30%)
					double_t Progress = 0.7 + ((double_t(c * iWidthStep * iHeightStep) + // 已经处理多少通道的像素数
						double_t(col * iHeightStep + row)) / // 当前通道已处理行列的像素数
						double_t(vImgChannels.size() * iWidthStep * iHeightStep)) * 0.3;// 总共像素数量
					ptrBridge->SetProgress(Progress);

					// 如果Flag为结束符号
					if (ptrBridge->StopFlag())
						return;
				}
			}

			vImgChannels_SR.push_back(imgC_SR);
		}

		// 通道合并
		merge(vImgChannels_SR, imgSR);

		// YUV转换为BGR(RGB)
		cvtColor(imgSR, imgSR, COLOR_YUV2BGR);

		// 截取矩形
		Rect rectSub = Rect(0, 0, iOriginWidth * 2, iOriginHeight * 2);

		if (bCutFlag)
		{
			// 裁剪
			imgSR = imgSR(rectSub);
		}

		// 保存图像
		imwrite(s_AlgorithmParm.strOutputPath, imgSR);

		// 完成信号
		ptrBridge->Accomplish();
	}
	catch(cv::Exception& e)
	{
		// 异常
		ptrBridge->operator<<(Log(FAILED_MODEL_LOAD, e.what()));

	}
}