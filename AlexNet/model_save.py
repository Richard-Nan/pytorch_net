import torch
import torchvision
from net import MyAlexNet
# vgg16 = torchvision.models.vgg16(pretrained = False)
# torch.save(vgg16,"vgg16_method1.pth")


# 读取 加载模型
model = torch.load("E:/myproject/AlexNet/save_model/best_model.pth")

model1 = MyAlexNet()

model1.load_state_dict(model)
print(model1)