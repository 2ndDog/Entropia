import torch
import RDDLN

PATH = ".\\checkpoint\\RDDLN_animeX2.pth"
OUTPATH = ".\\checkpoint\\"

batch_size = 1

# 部署模型
net = RDDLN.RDDLN()
net.load_state_dict(torch.load(PATH))
net.eval()

x = torch.randn(batch_size, 1, 64, 64, requires_grad=True)

torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  OUTPATH + "RDDLNx2_anime.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})