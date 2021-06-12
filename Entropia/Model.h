#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include "StatusValue.hpp"
#include "Algorithm.h"

/*
* ���Model���������ģ��(�ӿ�)
* ģ���ɴ���������ȥ
* ÿ���������һ��ģ���������
*/



class Model
{
public:
	Model();
	~Model();

	// ��ȡ�㷨�б�
	BOOL AlgorithmList(std::vector<std::shared_ptr<MAlgorithm>>& vctAlgorithmList);
	// ��ȡģ������
	std::string ModelName();
protected:
	// ģ������
	std::string strModelName;
	// ��ģ�͵��㷨��
	std::vector<std::shared_ptr<MAlgorithm>> vctAlgorithms;
private:
};


// ����һ����ȡ����ģ��vector�ĺ���
BOOL GlobalModelList(std::vector<std::shared_ptr<Model>>& TargetGlobalModelList);


// ���ֱ���ģ������
class RDDLN_SR :public Model
{
public:
	RDDLN_SR();
	~RDDLN_SR();

	
private:
};


// �޸�JPEGʧ��ģ������
class RDDLN_JPEG:public Model
{
public:
	RDDLN_JPEG();
	~RDDLN_JPEG();

	
private:

};
