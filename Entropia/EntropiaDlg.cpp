
// EntropiaDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "Entropia.h"
#include "EntropiaDlg.h"
#include "afxdialogex.h"

using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
	virtual BOOL OnInitDialog();
	//HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
public:
	void OnClickedLink();
	// 发布地址
	CStringW m_cstrLink;
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_LINK, m_cstrLink);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	ON_STN_CLICKED(IDC_LINK, &CAboutDlg::OnClickedLink)
	ON_WM_CTLCOLOR()
END_MESSAGE_MAP()

BOOL CAboutDlg::OnInitDialog()
{

	return TRUE;
}


HBRUSH CAboutDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);

	// TODO:  在此更改 DC 的任何特性
	if (pWnd->GetDlgCtrlID() == IDC_LINK)
	{
		pDC->SetTextColor(RGB(21, 62, 167));//用RGB宏改变颜色
	}

	// TODO:  如果默认的不是所需画笔，则返回另一个画笔
	return hbr;
}

void CAboutDlg::OnClickedLink()
{
	// TODO: 点击发布地址跳转网页
	ShellExecute(0, NULL, L"https://github.com/2ndDog/Entropia", NULL, NULL, SW_NORMAL);
}


// CEntropiaDlg 对话框



CEntropiaDlg::CEntropiaDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_ENTROPIA_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDI_ICON);
}

void CEntropiaDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_EXT, m_cbExtension);
	DDX_Control(pDX, IDC_COMBO_MOD, m_cbModel);
	DDX_Control(pDX, IDC_COMBO_SIZE, m_cbSize);
	DDX_Control(pDX, IDC_PROGRESS, m_ProgressCtrl);
	DDX_Control(pDX, IDC_ALGORITHM, m_cbAlgorithm);
	DDX_Text(pDX, IDC_EDIT_IN, s_IOParm.cstrInPathName);
	DDX_Text(pDX, IDC_EDIT_OUT, s_IOParm.cstrOutDirName);
	DDX_Text(pDX, IDC_EDIT_FILE, s_IOParm.cstrOutFileName);
	DDX_Text(pDX, IDC_LOG, m_cstrLog);
	DDX_Control(pDX, IDC_EXE, m_butExe);
	DDX_Control(pDX, IDC_CANCEL, m_butCancel);
	DDX_Control(pDX, IDC_CHECK_CUDA, m_cekCUDA);
}

BEGIN_MESSAGE_MAP(CEntropiaDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_SELECT_IN, &CEntropiaDlg::OnBnClickedSelectIn)
	ON_BN_CLICKED(IDC_SELECT_OUT, &CEntropiaDlg::OnBnClickedSelectOut)
	ON_CBN_SELCHANGE(IDC_COMBO_MOD, &CEntropiaDlg::OnSelchangeComboMod)
	ON_MESSAGE(WM_DISPLAY_LOG, DisplayLog)
	ON_MESSAGE(WM_STATUS_GUI, GUIStatus)
	ON_MESSAGE(WM_PROGRESS, RefreshProgress)
	ON_MESSAGE_VOID(WM_CLOSE, ReClose) //新添加的map
	ON_BN_CLICKED(IDC_EXE, &CEntropiaDlg::OnBnClickedExe)
	ON_BN_CLICKED(IDC_CANCEL, &CEntropiaDlg::OnBnClickedCancel)
END_MESSAGE_MAP()


// CEntropiaDlg 消息处理程序

BOOL CEntropiaDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// 窗口固定大小
	SetWindowLong(m_hWnd, GWL_STYLE, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX);

	// TODO: 在此添加额外的初始化代码
	// 获取内存信息
	statex.dwLength = sizeof(statex);
	GlobalMemoryStatusEx(&statex);
	// GUI状态标准
	iGUIStatus = STATUS_GUI_GENERAL;
	// 初始化桥接指针
	ptrBridge = make_shared<Bridge>();
	// Combo初始化
	m_cbExtension.SetCurSel(0);// Combo默认值
	m_cbSize.SetCurSel(0);// Combo默认值
	// 进度条初始化
	m_ProgressCtrl.SetRange(0, PROGRESS_MAX);// 进度条范围
	m_ProgressCtrl.SetPos(0);// 进度条位置
	InitModelList();
	// 启动状态线程
	m_pStatusThread = AfxBeginThread(StatusThread, (LPVOID)this);
	

	// 显示内存到日志上
	ptrBridge->operator<<(Log(0, "批量自动设置: 当前可用内存大小" + to_string(statex.ullAvailPhys / (1024 * 1024 * 1024)) + "GB"));

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CEntropiaDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CEntropiaDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CEntropiaDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

// 设置GUI输入输出的参数
void CEntropiaDlg::DisplayIOParm()
{
	SetDlgItemText(IDC_EDIT_IN, s_IOParm.cstrInPathName);
	SetDlgItemText(IDC_EDIT_OUT, s_IOParm.cstrOutDirName);
	//SetDlgItemText(IDC_EDIT_FILE, s_IOParm.cstrOutFileName);
}

// 根据输入自动设置输出参数
void CEntropiaDlg::AutoOutParm()
{
	UpdateData(TRUE);
	// 输入目录复制到输出
	s_IOParm.cstrOutDirName = s_IOParm.cstrInDirName;

	// 去掉扩展名
	LPWSTR lpsFileName = s_IOParm.cstrInFileName.GetBuffer();
	PathRemoveExtension(lpsFileName);

	// 添加后缀
	CStringW cstrCompleteFileName;
	cstrCompleteFileName.Format(L"%s%s", lpsFileName, L"_Entropia");



	// 复制到输出
	s_IOParm.cstrOutFileName = cstrCompleteFileName;
	UpdateData(FALSE);

	//DisplayIOParm();
}


void CEntropiaDlg::OnBnClickedSelectIn()
{
	// TODO: 选择输入文件
	// 
	// 
	// 打开文件的属性
	CStringW cstrFileAttributes = L"Image Files(*.JPEG;*.PNG;*.BMP;*.TIFF)|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff||";
	// TRUE为OPEN对话框，FALSE为SAVE AS对话框，“OFN_ALLOWMULTISELECT”可多选
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, cstrFileAttributes, NULL);

	if (dlg.DoModal() == IDOK)
	{
		UpdateData(TRUE);
		s_IOParm.cstrInPathName = dlg.GetPathName();// 存储打开文件的绝对路径
		s_IOParm.cstrInFileName = dlg.GetFileName();// 存储打开文件的文件名
		s_IOParm.cstrInDirName = dlg.GetFolderPath();// 存储打开文件的目录路径
		UpdateData(FALSE);

		// 自动设置输出参数
		AutoOutParm();
	}
	else
	{
		return;
	}
}


void CEntropiaDlg::OnBnClickedSelectOut()
{
	// TODO: 选择输出文件夹

	// 选择输出路径
	TCHAR cstrDir[MAX_PATH];
	BROWSEINFO bi;
	LPITEMIDLIST pidl;
	bi.hwndOwner = this->m_hWnd;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = cstrDir;// 输出缓冲区
	bi.lpszTitle = _T("请选择文件夹："); // 选择界面的标题  
	bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;// 新的界面
	bi.lpfn = NULL;
	bi.lParam = 0;
	bi.iImage = 0;
	pidl = SHBrowseForFolder(&bi);// 弹出对话框   
	if (pidl == NULL)// 点了取消，或者选择了无效的文件夹则返回NULL
	{
		return;
	}

	// 获取路径
	SHGetPathFromIDListW(pidl, cstrDir);

	// 设置输出路径
	UpdateData(TRUE);
	s_IOParm.cstrOutDirName = cstrDir;
	UpdateData(FALSE);
}

void CEntropiaDlg::DisplayProgress()
{
	// 显示进度,0~1映射到0~10'000
	m_ProgressCtrl.SetPos(ptrBridge->GetProgress());
}

void CEntropiaDlg::DisplayAlgorithm()
{

}

void CEntropiaDlg::InitModelList()
{
	
	GlobalModelList(vctModels);


	// 放入Combo控件中
	for (auto Model : vctModels)
	{
		m_cbModel.AddString(CStringW(Model->ModelName().c_str()));
	}
}



void CEntropiaDlg::OnSelchangeComboMod()
{
	// TODO: 选择模板Combo控件,更新算法Combo控件
	// 通过获取算法Vector来添加Combo内容
	// 清空MFC
	m_cbAlgorithm.ResetContent();
	// 获取当前Combo选择的模板位置
	int iCbTarget = m_cbModel.GetCurSel();
	if (iCbTarget == CB_ERR)
	{
		return;
	}
	// 获取当前Combo选择模板位置的字符
	CStringW cstrCbTarget;
	m_cbModel.GetLBText(iCbTarget, cstrCbTarget);
	// 判断是否与模板列表相等
	try
	{
		// 获取模型名称
		CStringW cstrModelName = CStringW(vctModels.at(iCbTarget)->ModelName().c_str());
		if (cstrModelName != cstrCbTarget)// 如果Combo选择的和实际的模型不匹配
		{
			MessageBox(L"模型名称不对应!", L"错误");
			return;
		}

		// 获取模型中的算法列表
		vctModels.at(iCbTarget)->AlgorithmList(vctAlgorithms);
		// 将算法列表添加到Combo中
		for (auto MAlgorithm : vctAlgorithms)
		{
			m_cbAlgorithm.AddString(CStringW(MAlgorithm->AlgorithmName().c_str()));
		}

		// 设置算法Combo后自动选择算法Combo
		if (m_cbAlgorithm.GetCount() > 0)// 如果Combo有值
		{
			m_cbAlgorithm.SetCurSel(0);// 默认选择第一个
		}
	}
	catch (const std::out_of_range& oor)
	{
		// vector.at()越界异常
		oor;

		return;
	}
}

LRESULT CEntropiaDlg::DisplayLog(WPARAM wParam, LPARAM lParam)
{
	SetDlgItemText(IDC_LOG, m_cstrLog);
	return 0;
}

LRESULT CEntropiaDlg::GUIStatus(WPARAM wParam, LPARAM lParam)
{
	// 如果状态没有改变,直接返回
	if (iGUIStatus == lParam)
	{
		return 0;
	}

	switch (lParam)
	{
	case STATUS_GUI_LOCK:
		// 开始不可用
		m_butExe.EnableWindow(0);
		// 取消可用
		m_butCancel.EnableWindow(1);
		break;
	case STATUS_GUI_GENERAL:
		// 开始可用
		m_butExe.EnableWindow(1);
		// 取消不可用
		m_butCancel.EnableWindow(0);
		break;
	default:
		break;
	}
	// 设置状态
	iGUIStatus = int(lParam);
	return 0;
}

LRESULT CEntropiaDlg::RefreshProgress(WPARAM wParam, LPARAM lParam)
{
	m_ProgressCtrl.SetPos(ptrBridge->GetProgress());
	return 0;
}

UINT CEntropiaDlg::StatusThread(LPVOID lpParam)
{
	// 获取传入的线程配置
	CEntropiaDlg* pCDlg = (CEntropiaDlg*)lpParam;
	// Log内容
	CStringW cstrLog;
	while (true)
	{
		// 判断是否结束
		if (pCDlg->iGUIStatus == STATUS_GUI_STOP)
		{
			return 0;
		}


		// 线程判断
		DWORD dwCode = 0;
		if (pCDlg->m_pAlgorithmThread != nullptr)
		{
			bool bRes = ::GetExitCodeThread(pCDlg->m_pAlgorithmThread->m_hThread, &dwCode);
		}
		if (dwCode != STILL_ACTIVE)// 算法线程结束(不存在)
		{
			// 开启GUI
			//线程向对话框发送自定义消息 
			if (pCDlg->iGUIStatus == STATUS_GUI_LOCK)
			{
				::PostMessage(pCDlg->m_hWnd, WM_STATUS_GUI, 0, STATUS_GUI_GENERAL);
			}
		}


		// 刷新进度条
		pCDlg->m_ProgressCtrl.SetPos(pCDlg->ptrBridge->GetProgress());

		// 日志操作
		// 如果队列为空返回不操作
		if (pCDlg->ptrBridge->queLog.empty())
		{
			continue;
		}
		// 获取日志内容
		cstrLog = pCDlg->ptrBridge->queLog.front().cstrLogContent;
		pCDlg->ptrBridge->queLog.pop();

		// 追加文本
		pCDlg->m_cstrLog += cstrLog;
		pCDlg->m_cstrLog += L"\r\n";// 加一个换行

		//线程向对话框发送自定义消息 
		::PostMessage(pCDlg->m_hWnd, WM_DISPLAY_LOG, 0, 0);

		Sleep(20);
	}

	return 0;
}


UINT CEntropiaDlg::ExeAlgorithm(LPVOID lpParam)
{
	// 获取传入的线程配置
	CEntropiaDlg* pCDlg = (CEntropiaDlg*)lpParam;
	// 父类指针
	MAlgorithm* pAlgorithm;
	try
	{
		// 获取当前选择算法的指针
		// shared_ptr<MAlgorithm> ptrAlgorithm = make_shared<MAlgorithm>(pCDlg->vctAlgorithms.at(pCDlg->m_cbAlgorithm.GetCurSel()));
		pAlgorithm = pCDlg->vctAlgorithms.at(pCDlg->m_cbAlgorithm.GetCurSel()).get();
	}
	catch (const std::out_of_range& oor)
	{
		// vector.at()越界异常
		oor;

		return -1;
	}

	pAlgorithm->Execute(pCDlg->ptrBridge, pCDlg->s_AlgorithmParm);

	return 0;
}


void CEntropiaDlg::OnBnClickedExe()
{
	// TODO: 点击开始
	// 设置参数
	// 检查参数
	// 多线程运行

	UpdateData(TRUE);
	UpdateData(FALSE);

	// 获取选择的后缀
	CStringW cstrExtension;
	m_cbExtension.GetLBText(m_cbExtension.GetCurSel(), cstrExtension);

	// 拼接最终输出文件
	s_IOParm.cstrOutPathName.Format(L"%s\\%s%s",
		s_IOParm.cstrOutDirName.GetBuffer(),
		s_IOParm.cstrOutFileName.GetBuffer(),
		cstrExtension.GetBuffer());

	// 检测输入输出设置是否为空
	if (s_IOParm.Empty())
	{
		MessageBox(L"输入输出未设置!", L"Failed!");
		return;
	}

	// 检测输出文件名是否合法
	if (!s_IOParm.IsFileNameValid())// 如果不合法
	{
		MessageBox(L"输出文件名不能包以下字符:\r\n\\ / : * \? \" < > |", L"Failed!");
		return;
	}

	// 文件名长度是否合法
	if (s_IOParm.cstrOutPathName.GetLength() > MAX_PATH)
	{
		MessageBox(L"文件名太长!", L"Failed!");
		return;
	}

	// 检测是否选择模型
	if (vctAlgorithms.size() == 0 || m_cbAlgorithm.GetCurSel() == CB_ERR)// 模型数量为0或没有选择
	{
		MessageBox(L"未选择模型算法!", L"Failed!");
		return;
	}

	// 准备执行算法
	s_AlgorithmParm.strInputPath = CW2A(s_IOParm.cstrInPathName);
	s_AlgorithmParm.strOutputPath = CW2A(s_IOParm.cstrOutPathName);

	// 根据可用内存设置Batch Size
	s_AlgorithmParm.iBatchSize = int(statex.ullTotalPhys / (1024 * 1024 * 1024));

	// 是否使用CUDA
	int iCUDA = m_cekCUDA.GetCheck();
	s_AlgorithmParm.bCUDA = (iCUDA == 1);

	// 启动多线程
	// 锁UI
	::PostMessage(this->m_hWnd, WM_STATUS_GUI, 0, STATUS_GUI_LOCK);
	ptrBridge->Run();
	m_pAlgorithmThread = AfxBeginThread(ExeAlgorithm, (LPVOID)this);
}


void CEntropiaDlg::OnBnClickedCancel()
{
	// TODO: 结束算法线程
	if (m_pAlgorithmThread->m_nThreadID == 0)
	{
		return;
	}
	//PostThreadMessage(m_pAlgorithmThread->m_nThreadID, WM_QUIT, 0, 0);

	ptrBridge->Stop();

	ptrBridge->operator<<(Log(0, "正在结束..."));
}


void CEntropiaDlg::ReClose()
{
	iGUIStatus = STATUS_GUI_STOP;

	this->OnClose();
}


