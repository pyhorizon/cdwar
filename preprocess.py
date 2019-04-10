import os
import shutil
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as ts
from matplotlib import pyplot as plt
from skimage import io
from PIL import Image as im
#firstly, fit the torch api
def to_torch(path):
    for c,i in enumerate(os.listdir(path)):
        if 'cat' in i:
            cat = str(c) + '.jpg'
            shutil.copy(path + '/' + i, './data_torch/cat/' + cat)
            #break
        else:
            dog = str(c) + '.jpg'
            shutil.copy(path + '/' + i, './data_torch/dog/' + dog)


def show(tensor):
    img=tensor.permute(2,1,0)
    plt.imshow(img)
    plt.show()

def prep(path,trans_list=[ts.Resize((255,255)),ts.CenterCrop(224),ts.ToTensor()]\
         ,batch_size=128,shuffle=True):
    transform = ts.Compose(trans_list)
    dataset=ImageFolder(path,transform=transform)
    data=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return data

def to_tensor_cuda(path):
    x=im.open(path)
    func=ts.ToTensor()
    func1=ts.Resize((224,224))
    img2tensor=func1(x)
    img2tensor=func(img2tensor)
    return img2tensor.view(-1,3,224,224).cuda()





if __name__=="__main__":
    data=prep('./data_torch',batch_size=10,trans_list=None)
    for i,j in data:
        show(i[0])
        print('this is a dog!') if j[0]==1 else print('this is a cat!')
        break