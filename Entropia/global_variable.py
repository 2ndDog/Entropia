progress = 0.0 # 进度条 0~1

logs = []

signal_termination = False

signal_running = 0 #运行状态 0为空闲


# 进度条类
def setProgress(variable):
    global progress
    progress = variable


def getProgress():
    global progress
    return progress


# 日志类
def setLogs(content, status):
    log = (content, status)
    logs.append(log)


def getLogs():
    temp_logs = logs.copy()
    logs.clear()
    return temp_logs


# 信号类
def Cancel():
    global signal_termination
    signal_termination = True # 中止信号true为中止


def Perform():
    global signal_termination
    signal_termination = False # 开始中止信号变为false

# 获取中止信号
def SignalTermination():
    global signal_termination
    return signal_termination


# 运行状态
def Running(id):
    global signal_running
    signal_running = id


# 结束运行
def Ending():
    global signal_running
    signal_running = 0


# 获取运行状态
def SignalRuning():
    global signal_running
    return signal_running
