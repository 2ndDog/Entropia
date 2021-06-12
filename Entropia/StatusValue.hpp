#pragma once
#include <string>

/***************************************常量定义*/
constexpr auto PROGRESS_MAX = 10000;// 进度条范围
/***************************************状态定义*/
constexpr auto STATUS_GUI_GENERAL = 0;// 解锁GUI
constexpr auto STATUS_GUI_LOCK = 1;// 锁定GUI
constexpr auto STATUS_GUI_STOP = 10;// 结束
/***************************************错误定义*/
constexpr auto IO_PARM_ENPTY = 3001;// 输入输出参数为空
constexpr auto ILLEGAL_OUT_FILE = 3002;// 输出文件不合法
constexpr auto NO_TARGET_ALGORITHM = 3003;// 未选择模型
constexpr auto FAILED_IMAGE_LOAD = 2001;// 图像加载错误
constexpr auto FAILED_MODEL_LOAD = 2002;// 模型加载错误


// 算法输入参数
struct AlgorithmParm
{
	std::string strInputPath;// 输入路径
	std::string strOutputPath;// 输出路径
	uint32_t iBatchSize;// 批处理大小
	bool bCUDA;// 是否使用CUDA
};


// 输出日志结构
struct Log
{
	int bLogID;//日志ID
	CStringW cstrLogContent;// 日志内容
	Log(int bLogID, CStringW cstrLogContent)
	{
		this->bLogID = bLogID;
		this->cstrLogContent = cstrLogContent;
	}
	Log(int bLogID, std::string cstrLogContent)
	{
		this->bLogID = bLogID;
		this->cstrLogContent = cstrLogContent.c_str();
	}
};

// 用于GUI与算法数据传输类
class Bridge
{
public:
	std::queue<Log> queLog;


	Bridge();
	~Bridge();
	
	// 重载<<运算符,写入Log
	void operator<<(const Log& log);
	// 设置进度条
	void SetProgress(double dProgress);
	// 返回进度条
	int GetProgress();
	// 开始
	void Run();
	// 运行中
	bool IsRun();
	// 结束
	void Stop();
	// 信号
	bool StopFlag();
	// 完成
	void Accomplish();
private:
	int iProgress;// 进度
	int iStatus;// 运行状态(未运行为0,正在运行为1)
	int iFlag;// 操作信号(由GUI设置,-1取消操作)
};

inline Bridge::Bridge()
{
	this->iProgress = 0;
	this->iStatus = 0;
	this->iFlag = 0;
}

inline Bridge::~Bridge()
{
}

inline void Bridge::operator<<(const Log& log)
{
	this->queLog.push(log);
}

inline void Bridge::SetProgress(double dProgress)
{
	// 0-1映射至0-MAX
	auto Range = [](int x) {x = x < 0 ? 0 : x; x = x > PROGRESS_MAX ? PROGRESS_MAX : x; return x; };
	// 设置进度
	this->iProgress = Range(int(dProgress * PROGRESS_MAX));
}

inline int Bridge::GetProgress()
{
	return iProgress;
}

inline void Bridge::Run()
{
	iStatus = 1;
	iFlag = 0;
}

inline bool Bridge::IsRun()
{
	if (iStatus == 1)
	{
		return true;
	}
	return false;
}

inline void Bridge::Stop()
{
	iFlag = -1;
}

inline bool Bridge::StopFlag()
{
	return iFlag == -1;
}

inline void Bridge::Accomplish()
{
	this->operator<<(Log(1, "完成"));
	iStatus = 0;
}

