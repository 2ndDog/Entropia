progress = 0.0

logs = []

signal_termination = False

signal_running = 0


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
    signal_termination = True


def Perform():
    global signal_termination
    signal_termination = False


def SignalTermination():
    global signal_termination
    return signal_termination


def Running(id):
    global signal_running
    signal_running = id


def Ending():
    global signal_running
    signal_running = 0


def SignalRuning():
    global signal_running
    return signal_running
