#include "pch.h"
#include "Model.h"

using namespace std;

// 模型的指针列表在这里
inline vector<shared_ptr<Model>> vctGlobalModelList{ make_shared<RDDLN_SR>(),make_shared<RDDLN_JPEG>() };

BOOL GlobalModelList(vector<shared_ptr<Model>>& TargetGlobalModelList)
{
    TargetGlobalModelList.assign(vctGlobalModelList.begin(), vctGlobalModelList.end());
    return 0;
}

Model::Model()
{
}

Model::~Model()
{
}

std::string Model::ModelName()
{
	return strModelName;
}

BOOL Model::AlgorithmList(vector<shared_ptr<MAlgorithm>>& vctAlgorithmList)
{
    // 先清空
    vctAlgorithmList.clear();
    // 拷贝
    vctAlgorithmList.assign(vctAlgorithms.begin(),vctAlgorithms.end());
	return 0;
}
