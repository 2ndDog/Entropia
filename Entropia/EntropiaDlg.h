
// EntropiaDlg.h: 头文件
//
#include <memory>
#include <queue>
#include <vector>
#include <thread>

#include "Model.h"

#pragma once

#define WM_STATUS_GUI (WM_USER + 100)
#define WM_PROGRESS (WM_USER + 200)
#define WM_DISPLAY_LOG (WM_USER + 300)


// 输入输出参数
struct IOParm
{
	CStringW cstrInPathName;// 输入文件的绝对路径(绑定DEIT)
	CStringW cstrInFileName;// 输入文件的文件名
	CStringW cstrInDirName;// 输入文件的目录路径
	CStringW cstrOutPathName;// 输出文件的绝对路径
	CStringW cstrOutFileName;// 输出文件的文件名(绑定DEIT)
	CStringW cstrOutDirName;// 输出文件的目录路径(绑定DEIT)

	// 判断路径是否为空
	bool Empty()
	{
		// 如果目录有一个为空则返回true表示为空
		return (cstrInPathName.IsEmpty() || cstrInFileName.IsEmpty() || cstrInDirName.IsEmpty() ||
			cstrOutPathName.IsEmpty() || cstrOutFileName.IsEmpty() || cstrOutDirName.IsEmpty());
	}

	// 判断文件名是否合法
	bool IsFileNameValid()
	{
		// 特殊字符检测
		WCHAR SpecialCha[] = { L'\\',L'/',L':',L'*',L'\?',L'\"',L'<',L'>',L'|' };
		for (int i = 0; i < sizeof(SpecialCha) / sizeof(WCHAR); i++)
		{
			if (NULL != _tcschr(cstrOutFileName, SpecialCha[i]))
			{
				return false;// 如果找到特殊字符则返回
			}
		}
		return true;
	}
};

// CEntropiaDlg 对话框
class CEntropiaDlg : public CDialogEx
{
// 构造
public:
	CEntropiaDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ENTROPIA_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;
	// 输入输出参数,负责存储和处理用户设置的输入和输出参数
	IOParm s_IOParm;
	// 算法输入参数,调用算法时,输入到Execute()函数的参数
	AlgorithmParm s_AlgorithmParm;
	// 内存信息,获取计算机内存,便于设置自动Batch
	MEMORYSTATUSEX statex;
	// GUI状态,作为控制控件的信号
	int iGUIStatus;
	// 模型列表,存储模型
	std::vector<std::shared_ptr<Model>> vctModels;
	// 算法列表,存储当前模型拥有的算法
	std::vector<std::shared_ptr<MAlgorithm>> vctAlgorithms;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	void DisplayIOParm();
	void AutoOutParm();
	DECLARE_MESSAGE_MAP();
public:
	// 扩展名控件
	CComboBox m_cbExtension;
	// 模型控件
	CComboBox m_cbModel;
	// 模型的算法控件
	CComboBox m_cbAlgorithm;
	// 批处理大小控件
	CComboBox m_cbSize;
	// 进度条控件
	CProgressCtrl m_ProgressCtrl;
	// 日志内容控件
	CStringW m_cstrLog;
	// 开始执行按钮
	CButton m_butExe;
	// 取消按钮
	CButton m_butCancel;
	// 线程信息
	CWinThread* m_pStatusThread;// 运行状态多线程,负责监控GUI和后台运行状态,并处理杂项
	CWinThread* m_pAlgorithmThread;// 算法线程,调用此算法后台处理图像
	// 桥接结构体指针,用于GUI和后台算法之间的状态信息传递
	std::shared_ptr<Bridge> ptrBridge;

	// 选择输入按钮
	afx_msg void OnBnClickedSelectIn();
	// 选择输出按钮
	afx_msg void OnBnClickedSelectOut();
	// 显示进度条
	afx_msg void DisplayProgress();
	// 显示算法
	afx_msg void DisplayAlgorithm();
	// 选择模型时的函数
	afx_msg void OnSelchangeComboMod();
	// 开始执行
	afx_msg void OnBnClickedExe();
	// 显示Log
	afx_msg LRESULT DisplayLog(WPARAM wParam, LPARAM lParam);
	// GUI
	afx_msg LRESULT GUIStatus(WPARAM wParam, LPARAM lParam);
	// 刷新进度条
	afx_msg LRESULT RefreshProgress(WPARAM wParam, LPARAM lParam);
	// 后台数据处理线程
	static UINT StatusThread(LPVOID lpParam);
	// 执行算法线程
	static UINT ExeAlgorithm(LPVOID lpParam);
	// 初始化模型列表
	void InitModelList();
	afx_msg void OnBnClickedCancel();
	afx_msg void ReClose();
	// 是否使用CUDA
	CButton m_cekCUDA;
};
