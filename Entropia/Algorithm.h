#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <opencv.hpp>

#include "StatusValue.hpp"

/*
* ���MAlgorithm�������㷨�ӿ�
* �㷨�ɴ���������ȥ
*/
class MAlgorithm
{
public:
	MAlgorithm();
	~MAlgorithm();

	virtual BOOL Execute(std::shared_ptr<Bridge> ptrBridge, AlgorithmParm s_AlgorithmParm) = 0;

	// �����㷨��
	std::string AlgorithmName();
protected:
	// �㷨��
	std::string strAlgorithmName;
private:
};
