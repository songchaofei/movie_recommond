import torch
import torch.nn as nn

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")


print(torch.version.cuda)  # 打印CUDA版本
print(torch.cuda.is_available())  # 检查是否可用
print(torch.backends.cudnn.enabled)  # 检查cuDNN是否可用


print(torch.cuda.device_count())  # 返回可用的GPU设备数量
print(torch.cuda.get_device_name(0))  # 返回第一个GPU设备的名称


#定义一个device对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)