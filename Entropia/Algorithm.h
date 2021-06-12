#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <opencv.hpp>

#include "StatusValue.hpp"

/*
* 这个MAlgorithm类用于算法接口
* 算法由此类派生出去
*/
class MAlgorithm
{
public:
	MAlgorithm();
	~MAlgorithm();

	virtual BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm) = 0;

	// 返回算法名
	std::string AlgorithmName();
protected:
	// 算法名
	std::string strAlgorithmName;
private:
};
