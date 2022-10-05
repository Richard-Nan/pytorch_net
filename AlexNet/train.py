import torch
from torch import nn
from net import MyAlexNet
from torch.optim import lr_scheduler
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


ROOT_TRAIN = "E:/myproject/AlexNet/data/train"
ROOT_TEST = 'E:/myproject/AlexNet/data/val'

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 将图像的像素值归一化为-1 1之间
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

# 改变图像的工具箱
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),  # 转化为Tensor类型
    normalize
])

var_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

train_dataset = ImageFolder(ROOT_TRAIN,transform=train_transform)

val_dataset = ImageFolder(ROOT_TEST,transform = var_transform)

train_dataloader = DataLoader(train_dataset,batch_size = 4,shuffle =True)
val_dataloader = DataLoader(val_dataset,batch_size = 4,shuffle =True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyAlexNet().to(device)
# 定义损失函数

loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 优化器

# 学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 定义训练函数

def train(dataloader,model,loss_fn,optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch,(x,y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)
        # 将数据转到GPU 上
        output = model(image)  # 预测值和真实值
        cur_loss = loss_fn(output, y)
        _,pred = torch.max(output, axis=1)  # 找最大值的坐标
        cur_acc =torch.sum(y == pred) / output.shape[0]  # 对比每个坐标的值 相等就是预测正确的 shape[0]表示这一组的预测个数

        # 反向传播
        optimizer.zero_grad()   # 反向传播的几个步骤
        cur_loss.backward()
        optimizer.step()    # 进入下一步
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print("train_loss：" + str(train_loss))  # 每批次的平均loss
    print("train_acc："+str(train_acc))  # 每批次的正确率
    return train_loss,train_acc

def val(dataloader,model,loss_fn):
    # 将模型转化为验证模式
    model.eval()   # 不计算梯度 转为验证模式
    with torch.no_grad():  # 注意这个括号 别忘带了  不计算梯度
        loss, current, n = 0.0, 0.0, 0
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)  # 预测值和真实值
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
    val_loss = loss / n
    val_acc = current / n

    print("val_loss：" + str(val_loss))  # 每批次的平均loss
    print("val_acc：" + str(val_acc))  # 每批次的正确率
    return val_loss, val_acc

# 定义画图函数
def matplot_loss(train_loss, val_loss):

    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()

def matplot_acc(train_acc, val_acc):
    # 画个图像
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()

#  开始训练
if __name__ == '__main__':
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val =[]
    epoch = 100
    min_acc = 0
    for t in range(epoch):
        lr_scheduler.step()
        print(f"epoch{t+1}\n...........")
        train_loss ,train_acc= train(train_dataloader,model,loss_fn, optimizer)
        val_loss,val_acc = val(val_dataloader,model,loss_fn)
        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)
        # 保存最好的模型权重
        if val_acc > min_acc:
            folder = "save_model"
            if not os.path.exists(folder):
                os.mkdir("save_model")
            min_acc = val_acc
            print(f"保持最好的模型，第{t+1}轮")
            torch.save(model.state_dict(),"save_model/best_model.pth")
            # 保存最后一轮的模型
        if t == epoch + 1:
            torch.save(model.state_dict(),"save_model/last_model.pth")
    matplot_loss(loss_train, loss_val)
    matplot_acc(acc_train, acc_val)