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
	// ��CStringת��Ϊstring
	string strInPath = string(CW2A(cstrInPath));

	// ��ȡͼ��
	Mat imgIn = imread(strInPath);

	// �����ͼ��Ϊ�շ���
	if (imgIn.empty())
	{
		return FAILED_IMAGE_LOAD;
	}

	// ���سɹ���ֵ��
	this->imgOrigin = imgIn;
	return 0;
}
