#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include "StatusValue.hpp"
#include "Algorithm.h"

/*
* 这个Model类管理所有模型(接口)
* 模型由此类派生出去
* 每个子类代表一个模型依次添加
*/



class Model
{
public:
	Model();
	~Model();

	// 获取算法列表
	BOOL AlgorithmList(std::vector<std::shared_ptr<MAlgorithm>>& vctAlgorithmList);
	// 获取模型名称
	std::string ModelName();
protected:
	// 模型名称
	std::string strModelName;
	// 该模型的算法组
	std::vector<std::shared_ptr<MAlgorithm>> vctAlgorithms;
private:
};


// 这是一个获取所有模型vector的函数
BOOL GlobalModelList(std::vector<std::shared_ptr<Model>>& TargetGlobalModelList);


// 超分辨率模型子类
class RDDLN_SR :public Model
{
public:
	RDDLN_SR();
	~RDDLN_SR();

	
private:
};


// 修复JPEG失真模型子类
class RDDLN_JPEG:public Model
{
public:
	RDDLN_JPEG();
	~RDDLN_JPEG();

	
private:

};
