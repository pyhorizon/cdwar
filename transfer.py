import torch
import preprocess as ps
import matplotlib.pyplot as plt
import torchvision.models as model
import os
from torch import optim
#firstly, load model resnet
from torch.autograd import Variable
batch_size=128
epoch=range(3)
class tr_net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(2048,1024)
        self.l2 = torch.nn.Linear(1024,500)
        self.l3 = torch.nn.Linear(500,2)
        self.r = torch.nn.ReLU()
        self.c = torch.nn.Softmax(dim = 1)
    def forward(self, x):
        y1=self.r(self.l1(x))
        y2=self.r(self.l2(y1))
        y3=self.c(self.l3(y2))
        return y3
my_net=tr_net().cuda()
net=model.resnet50(pretrained=True).cuda()
for i in net.parameters():
    i.requires_grad = False

net.fc=my_net

data=ps.prep('./data_torch',batch_size=batch_size)
loss_f=torch.nn.CrossEntropyLoss()
optimizer=optim.Adam(net.fc.parameters())
for e in epoch:
    for i,(x,y) in enumerate(data):
        x = Variable(x).cuda()
        label = Variable(y).cuda()
        y_pre=net.forward(x)
        error=loss_f(y_pre,label)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        if i % 10 ==0:
            print('epoch:',e,'batch:',i)
            #print(error.data)
            print('acc:',torch.sum(torch.argmax(y_pre,1)==label).type(torch.float64)/batch_size)
            #print(torch.sum(label))

torch.save(net,'./dog_cat_model.pkl')

dog_cat_net=torch.load('./dog_cat_model.pkl')
dog_cat_net.eval()
print(dog_cat_net.forward(ps.to_tensor('./test1/11.jpg')))
if torch.argmax(dog_cat_net.forward(ps.to_tensor('./test1/11.jpg')),1)==0:
    print(torch.argmax(dog_cat_net.forward(ps.to_tensor('./test1/11.jpg')),1))
    print('cat')
else:
    print('dog')


