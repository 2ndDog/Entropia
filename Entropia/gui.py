from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter.scrolledtext import ScrolledText
from tkinter import END
import tkinter.messagebox
import os
import _thread
import time
import re
import global_variable as Gvar
import enlarge as enl
import denoise as noc


# 选择图片
def SelectIenle():
    file_name_extension = "*.jpeg;*.jpg;*.jp2;*.png;*.bmp;*.dib;*.tiff;*.tif"
    path_input = askopenfilename(title="选择图片",
                                 filetypes=[("Ienle file", file_name_extension)
                                            ])

    # 选择
    if path_input != "":
        # 当前文件绝对路径
        path_input = os.path.abspath(path_input)
        # 自动获取加后缀的输出文件路径
        path_output = os.path.join(
            os.path.abspath(os.path.dirname(path_input)),
            AddExtension(path_input))

        field_path_input.set(path_input)
        field_path_output.set(path_output)


# 选择路径
def SelectPath():
    path_output = askdirectory(title="选择保存的路径")

    # 选择
    if path_output != "":
        # 获取加后缀的输出文件路径
        path_output = os.path.join(os.path.abspath(path_output),
                                   AddExtension(field_path_input.get()))

        field_path_output.set(path_output)


# IO功能类
def FunctionIO(frame_parent):
    frame_parent.grid_columnconfigure(1, weight=1)

    #
    lab_input = Label(frame_parent, text="输入路径")
    lab_input.grid(row=0, column=0, sticky=W, padx=pads)
    #
    lab_output = Label(frame_parent, text="输出路径")
    lab_output.grid(row=1, column=0, sticky=W, padx=pads)

    # 文本框
    ent_input = Entry(frame_parent, textvariable=field_path_input)
    ent_input.grid(row=0, column=1, sticky=W + E)
    # 文本框
    ent_output = Entry(frame_parent, textvariable=field_path_output)
    ent_output.grid(row=1, column=1, sticky=W + E)

    # 浏览按钮
    but_input = Button(frame_parent, text="浏览...", command=SelectIenle)
    but_input.grid(row=0, column=3, sticky=E, padx=pads)
    # 浏览按钮
    but_output = Button(frame_parent, text="浏览...", command=SelectPath)
    but_output.grid(row=1, column=3, sticky=E, padx=pads)


# 配置模式
def ModeConf(frame_parent):
    # 放大选择按钮
    rad_mode_enl = Radiobutton(frame_parent,
                               text="放大",
                               variable=var_mode,
                               value=1,
                               command=lambda: FunctionConf(frame_conf))
    rad_mode_enl.grid(row=0, column=0)

    # 降噪选择按钮
    rad_mode_dst = Radiobutton(frame_parent,
                               text="降噪",
                               variable=var_mode,
                               value=2,
                               command=lambda: FunctionConf(frame_conf))
    rad_mode_dst.grid(row=1, column=0)


# 放大
def Enlarge(frame_parent):
    # 放大倍率
    rad_enl_mul = Radiobutton(frame_parent,
                              text="放大倍率",
                              variable=var_enl,
                              value=1,
                              command=lambda: Enlarge(frame_parent))
    rad_enl_mul.grid(row=0, column=0, sticky=W)

    # 放大后尺寸
    rad_enl_size = Radiobutton(frame_parent,
                               text="放大后尺寸",
                               variable=var_enl,
                               value=2,
                               command=lambda: Enlarge(frame_parent))
    rad_enl_size.grid(row=1, column=0, sticky=W)

    # 倍率
    ent_mul = Entry(frame_parent,
                    show=None,
                    textvariable=var_mul,
                    font=('Arial', 8))
    ent_mul.grid(row=0, column=1, padx=pads)
    # 尺寸
    ent_size = Entry(frame_parent,
                     show=None,
                     textvariable=var_size,
                     font=('Arial', 8))
    ent_size.grid(row=1, column=1, padx=pads)

    if var_mode.get() != 1:
        rad_enl_mul.configure(state="disabled")
        rad_enl_size.configure(state="disabled")
        ent_mul.configure(state="disabled")
        ent_size.configure(state="disabled")
    else:
        rad_enl_mul.configure(state="normal")
        rad_enl_size.configure(state="normal")
        ent_mul.configure(state="disabled")
        ent_size.configure(state="disabled")
        if var_enl.get() == 1:
            ent_mul.configure(state="normal")
        if var_enl.get() == 2:
            ent_size.configure(state="normal")

    # 更新后缀
    UpdateSuffix()


# 降噪等级
def DenoiseLevel(frame_parent):
    rad_lv1 = Radiobutton(frame_parent, text="1级", variable=var_noise, value=1,command=lambda: DenoiseLevel(frame_parent))
    rad_lv1.grid(row=0, column=0, sticky=W)

    rad_lv2 = Radiobutton(frame_parent, text="2级", variable=var_noise, value=2,command=lambda: DenoiseLevel(frame_parent))
    rad_lv2.grid(row=1, column=0, sticky=W)

    if var_mode.get() != 2:
        rad_lv1.configure(state="disabled")
        rad_lv2.configure(state="disabled")
    else:
        rad_lv1.configure(state="normal")
        rad_lv2.configure(state="normal")

    # 更新后缀
    UpdateSuffix()

'''

# 更新后缀
def UpdateSuffix():
    while (True):
        time.sleep(0.2)
        if field_path_input.get() == "":
            continue
        # 获取输出文件名
        path_output = field_path_output.get()
        # 分离路径和文件名
        (filepath, tempfilename) = os.path.split(path_output)
        path_output = os.path.join(os.path.abspath(filepath),
                                   AddExtension(field_path_input.get()))
        field_path_output.set(path_output)

'''


# 更新后缀
def UpdateSuffix():
    if field_path_input.get() == "":
        return 0
    # 获取输出文件名
    path_output = field_path_output.get()
    # 分离路径和文件名
    (filepath, tempfilename) = os.path.split(path_output)
    # 添加后缀
    path_output = os.path.join(os.path.abspath(filepath),
                               AddExtension(field_path_input.get()))
    field_path_output.set(path_output)


# 配置功能类
def FunctionConf(frame_parent):
    # 超分辨率模式
    frame_mode = LabelFrame(frame_parent, text="超分辨率模式")
    frame_mode.grid(row=0, column=0, columnspan=2, rowspan=2, padx=pads)
    ModeConf(frame_mode)

    # 放大尺寸
    frame_enlnify = LabelFrame(frame_parent, text="放大尺寸")
    frame_enlnify.grid(row=0, column=2, columnspan=2, rowspan=2)
    Enlarge(frame_enlnify)

    # 降噪等级
    frame_noise = LabelFrame(frame_parent, text="降噪等级")
    frame_noise.grid(row=0, column=4, rowspan=2, padx=pads)
    DenoiseLevel(frame_noise)

    # 执行
    but_execute = Button(frame_parent, text="开始", command=Perform)
    but_execute.grid(row=0, column=5, sticky=N)
    but_cancel = Button(frame_parent, text="取消", command=Cancel)
    but_cancel.grid(row=1, column=5, sticky=S)

    if var_state.get() == 0:
        but_execute.configure(state="normal")
        but_cancel.configure(state="disabled")
    if var_state.get() == 1:
        but_execute.configure(state="disabled")
        but_cancel.configure(state="normal")

    # 更新后缀
    UpdateSuffix()


# 显示进度
def Progress(frame_parent, bar_height):
    while (True):
        time.sleep(0.2)
        progress = Gvar.getProgress() * 474
        # 进度
        fill_bar = frame_parent.create_rectangle(0,
                                                 0,
                                                 progress,
                                                 bar_height,
                                                 fill="#1ABC9C",
                                                 outline="#1ABC9C")

        if Gvar.SignalRuning() == 0:
            frame_parent.delete("all")
            Gvar.setProgress(0.0)


# 进度条类
def ProgressBar(frame_parent):
    # 绘制空条
    bar_width = 100
    bar_height = 25
    empty_bar = Canvas(frame_parent,
                       width=bar_width,
                       height=bar_height,
                       bg="white")

    empty_bar.pack(fill=X)
    _thread.start_new_thread(Progress, (
        empty_bar,
        bar_height,
    ))


# 显示日志
def ShowLog(text_parent):
    while (True):
        time.sleep(0.2)
        # 获取日志
        logs = Gvar.getLogs()
        # 日志标签颜色
        text_parent.tag_config(1, foreground="#DC7633")  # 橙色警告
        text_parent.tag_config(2, foreground="#E74C3C")  # 红色错误

        # 打印标日志
        for num_log in range(len(logs)):
            # 解锁窗口
            text_parent.configure(state="normal")

            log = logs[num_log]
            text_parent.insert(END, log[0] + "\n", log[1])
            # 锁定窗口
            text_parent.configure(state="disabled")


# 日志类
def RunningLog(frame_parent):
    #
    text = ScrolledText(frame_parent)
    text.configure(state="disabled")
    text.pack(fill=BOTH)
    _thread.start_new_thread(ShowLog, (text, ))


# 检测是否修改文本框
def MonitorText():
    before_text_mul=var_mul.get() # 获取之前放大倍率
    before_text_size=var_size.get() # 获取之前放大尺寸
    while(True):
        time.sleep(0.2)
        current_text_mul=var_mul.get() # 获取当前放大倍率
        current_text_size=var_size.get() # 获取当前放大尺寸

        # 检测是否更改文本
        if current_text_mul==before_text_mul and current_text_size==before_text_size:
            # 检测到未更改
            continue

        # 检测到更改
        UpdateSuffix()
        # 更新检测状态
        before_text_mul=current_text_mul
        before_text_size=current_text_size


# 添加后缀
def AddExtension(path_input):
    # 后缀名
    suffix_model = ""
    suffix_mode = ""
    suffix_value = ""

    # 放大
    if var_mode.get() == 1:
        suffix_mode = "(scale)"
        if var_enl.get() == 1:
            suffix_value = "(%.4f)" % GetMulValue(var_mul.get())
        else:
            temp_size = GetSizeValue(var_size.get())
            suffix_value = "(%dx%d)" % (temp_size[0], temp_size[1])
    # 降噪
    if var_mode.get() == 2:
        suffix_mode = "(noise)"
        suffix_value = "(level%d)" % var_noise.get()

    suffix = suffix_model + suffix_mode + suffix_value

    # 分离路径和文件名
    (filepath, tempfilename) = os.path.split(path_input)
    # 分离文件名和扩展名
    (filename, extension) = os.path.splitext(tempfilename)
    # 合成新文件名
    filename_new = filename + suffix + ".png"

    return filename_new


# 取消
def Cancel():
    Gvar.Cancel()
    Gvar.setLogs("中止中...", 1)


# 获取放大倍率
def GetMulValue(state_mul):
    state_mul = state_mul[:min(len(state_mul), 6)]  # 截取前6位
    list_temp_mul = re.findall(r"\d+\.?\d*", state_mul)  # 提取小数
    # 是否含有浮点数
    if len(list_temp_mul) < 1:
        return 0

    # 返回倍率
    return float(list_temp_mul[0])


# 放大尺寸
def GetSizeValue(state_size):
    state_size = state_size[:min(len(state_size), 10)]  # 截取前10位
    list_temp_size = re.findall(r"\d+", state_size)  # 提取正整数
    # 是否含有两个整数
    if len(list_temp_size) < 2:
        return None

    # 返回尺寸
    return (int(list_temp_size[0]), int(list_temp_size[1]))


# 执行
def Perform():
    # 获取当前设置状态
    path_iamge_input = field_path_input.get()
    path_iamge_output = field_path_output.get()
    state_mode = var_mode.get()
    state_enl = var_enl.get()
    state_mul = var_mul.get()
    state_size = var_size.get()
    state_noise = var_noise.get()

    if path_iamge_input == "":
        tkinter.messagebox.showerror(title="错误", message="缺少图片路径")
        return 1

    if path_iamge_output == "":
        tkinter.messagebox.showerror(title="错误", message="缺少图片路径")
        return 1

    if state_mode == 1:  # 放大模式

        # 放大倍率
        if state_enl == 1:
            # 获取浮点数
            temp_mul = GetMulValue(state_mul)

            # 是否含有浮点数
            if temp_mul == 0:
                tkinter.messagebox.showerror(title="错误", message="放大倍率无效输入!")
                return 1

            if (temp_mul < 1 and temp_mul > 4):
                tkinter.messagebox.showwarning(title="警告", message="建议放大倍率1~4")
                return 1

            _thread.start_new_thread(enl.Enlarge, (
                path_iamge_input,
                path_iamge_output,
                temp_mul,
                None,
            ))

        # 放大到尺寸
        if state_enl == 2:
            # 获取尺寸
            temp_size = GetSizeValue(state_size)

            if temp_size == None:
                tkinter.messagebox.showerror(title="错误", message="放大尺寸无效输入!")
                return 1

            if (temp_size[0] > 4000 or temp_size[0] > 4000):
                tkinter.messagebox.showwarning(title="警告",
                                               message="建议尺寸不超过4000x4000")
                return 1

            _thread.start_new_thread(enl.Enlarge, (
                path_iamge_input,
                path_iamge_output,
                None,
                temp_size,
            ))

    if state_mode == 2:  #降噪模式
        _thread.start_new_thread(noc.NoiseCancelling, (
            path_iamge_input,
            path_iamge_output,
            state_noise,
        ))

    # 开始信号
    Gvar.Perform()
    var_state.set(1)

    # 刷新GUI
    FunctionConf(frame_conf)


# 线程监视
def Monitor():
    while (True):
        time.sleep(0.5)
        if Gvar.SignalRuning() < var_state.get():  # 未运行或运行结束
            var_state.set(0)

            # 刷新GUI
            FunctionConf(frame_conf)


# 定义TK
window = Tk()

field_path_input = StringVar()  # 输入路径
field_path_output = StringVar()  # 输出路径
var_mode = IntVar(value=1)  # 超分辨率模式
var_enl = IntVar(value=1)  # 放大模式
var_mul = StringVar(value=1.0000)
var_size = StringVar(value="1920x1080")
var_noise = IntVar(value=1)  # 降噪等级
var_state = IntVar(value=0)  # 运行状态

# 标题
window.title("Entropia")

# 默认字体
window.option_add("*Font", "思源黑体light 10")

# 添加图标
window.iconbitmap("Entropia.ico")

# 窗口属性
# 窗口大小(高 * 宽)
width = 500
height = 300
#获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
alignstr = "%dx%d+%d+%d" % (width, height, (screenwidth - width) // 2,
                            (screenheight - height) // 2)
window.geometry(alignstr)
# 防止用户调整尺寸
window.resizable(height=False, width=False)

pads = 8

# 框架布局
# 主框架
frame_root = Frame(window).pack(fill=BOTH, padx=pads * 4, pady=pads * 0.5)
# 功能框架
frame_io = LabelFrame(frame_root, text="输入和输出设置")
frame_conf = LabelFrame(frame_root, text="处理设置")
frame_progress = Frame(frame_root)
frame_log = Frame(frame_root)

frame_io.pack(fill=X, padx=pads * 1.4, pady=pads * 0.5, ipady=pads * 0.5)
frame_conf.pack(fill=X, padx=pads * 1.4, ipadx=pads * 0.5, ipady=pads * 0.5)
frame_progress.pack(fill=X, padx=pads * 1.4)
frame_log.pack(fill=BOTH, padx=pads * 1.4, pady=pads * 1.4)

FunctionIO(frame_io)
FunctionConf(frame_conf)
ProgressBar(frame_progress)
RunningLog(frame_log)

# 监视者
_thread.start_new_thread(Monitor, ())

# 更新路径
_thread.start_new_thread(MonitorText, ())

# 循环
window.mainloop()
