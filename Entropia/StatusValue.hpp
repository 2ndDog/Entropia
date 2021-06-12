#pragma once
#include <string>

/***************************************��������*/
constexpr auto PROGRESS_MAX = 10000;// ��������Χ
/***************************************״̬����*/
constexpr auto STATUS_GUI_GENERAL = 0;// ����GUI
constexpr auto STATUS_GUI_LOCK = 1;// ����GUI
constexpr auto STATUS_GUI_STOP = 10;// ����
/***************************************������*/
constexpr auto IO_PARM_ENPTY = 3001;// �����������Ϊ��
constexpr auto ILLEGAL_OUT_FILE = 3002;// ����ļ����Ϸ�
constexpr auto NO_TARGET_ALGORITHM = 3003;// δѡ��ģ��
constexpr auto FAILED_IMAGE_LOAD = 2001;// ͼ����ش���
constexpr auto FAILED_MODEL_LOAD = 2002;// ģ�ͼ��ش���


// �㷨�������
struct AlgorithmParm
{
	std::string strInputPath;// ����·��
	std::string strOutputPath;// ���·��
	uint32_t iBatchSize;// �������С
	bool bCUDA;// �Ƿ�ʹ��CUDA
};


// �����־�ṹ
struct Log
{
	int bLogID;//��־ID
	CStringW cstrLogContent;// ��־����
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

// ����GUI���㷨���ݴ�����
class Bridge
{
public:
	std::queue<Log> queLog;


	Bridge();
	~Bridge();
	
	// ����<<�����,д��Log
	void operator<<(const Log& log);
	// ���ý�����
	void SetProgress(double dProgress);
	// ���ؽ�����
	int GetProgress();
	// ��ʼ
	void Run();
	// ������
	bool IsRun();
	// ����
	void Stop();
	// �ź�
	bool StopFlag();
	// ���
	void Accomplish();
private:
	int iProgress;// ����
	int iStatus;// ����״̬(δ����Ϊ0,��������Ϊ1)
	int iFlag;// �����ź�(��GUI����,-1ȡ������)
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
	// 0-1ӳ����0-MAX
	auto Range = [](int x) {x = x < 0 ? 0 : x; x = x > PROGRESS_MAX ? PROGRESS_MAX : x; return x; };
	// ���ý���
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
	this->operator<<(Log(1, "���"));
	iStatus = 0;
}

