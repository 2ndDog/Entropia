#include "pch.h"
#include "BasicImage.h"
#include "StatusValue.hpp"

using namespace std;
using namespace cv;


BasicImage::BasicImage()
{
}

BasicImage::~BasicImage()
{
}

BOOL BasicImage::LoadImage(CStringW cstrInPath)
{
	// 将CString转化为string
	string strInPath = string(CW2A(cstrInPath));

	// 读取图像
	Mat imgIn = imread(strInPath);

	// 如果该图像为空返回
	if (imgIn.empty())
	{
		return FAILED_IMAGE_LOAD;
	}

	// 加载成功则赋值到
	this->imgOrigin = imgIn;
	return 0;
}
