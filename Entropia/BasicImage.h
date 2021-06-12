#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <opencv.hpp>

class BasicImage
{
public:
	BasicImage();
	~BasicImage();

	// ╪стьм╪оЯ
	BOOL LoadImage(CStringW cstrInPath);
private:
	cv::Mat imgOrigin;
};
