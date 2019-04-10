#encoding:utf-8
#author:@pyhorizon Dingchang Sun 2269254303@qq.com
#using resnet18
#using cuda
#version:1.0
import torch
import preprocess as ps
import matplotlib.pyplot as plt
import torchvision.models as model
from torch import optim
from torch.autograd import Variable
from skimage import io
batch_size = 128
epoch = range(3)
class tr_net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(512,256)
        self.l2 = torch.nn.Linear(256,100)
        self.l3 = torch.nn.Linear(100,2)
        self.r = torch.nn.ReLU()
        self.c = torch.nn.Softmax(dim = 1)
    def forward(self, x):
        y1 = self.r(self.l1(x))
        y2 = self.r(self.l2(y1))
        y3 = self.c(self.l3(y2))
        return y3
my_net = tr_net().cuda()
net = model.resnet18(pretrained=True).cuda()
for i in net.parameters():
    i.requires_grad = False

net.fc = my_net
'''
data = ps.prep('./data_torch',batch_size=batch_size)
loss_f = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.fc.parameters())
for e in epoch:
    for i,(x,y) in enumerate(data):
        x = Variable(x).cuda()
        label = Variable(y).cuda()
        y_pre = net.forward(x)
        error = loss_f(y_pre,label)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        if i % 10==0:
            print('epoch:',e,'batch:',i)
            #print(error.data)
            print('acc:',torch.sum(torch.argmax(y_pre,1)==label).type(torch.float64)/batch_size)
            #print(torch.sum(label))

torch.save(net,'./dog_cat_model2.pkl')
'''
dog_cat_net = torch.load('./dog_cat_model.pkl')
dog_cat_net.eval()
path='./test1/144.jpg'
plt.imshow(io.imread(path))

print((dog_cat_net.forward(ps.to_tensor_cuda(path))).data)
if torch.argmax(dog_cat_net.forward(ps.to_tensor_cuda(path)),1)==0:
    plt.text(50,50,'cat:',fontsize=24,color='g')
    plt.text(150, 50, 'miao', fontsize=24, color='g')
else:
    plt.text(50,50,'dog:',fontsize=24,color='r')
    plt.text(150, 50, 'wang', fontsize=24, color='r')
plt.show()

