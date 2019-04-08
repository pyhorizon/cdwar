import os
import shutil
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as ts
from matplotlib import pyplot as plt
from skimage import io
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

def to_tensor(x):
    img=io.imread(x)
    img.resize(224,224,3)
    img=img.reshape(1,224,224,3)
    img2tensor=torch.from_numpy(img).type(torch.float32).cuda()
    img2tensor=img2tensor.permute(0,3,1,2)
    return img2tensor





if __name__=="__main__":
    data=prep('./data_torch',batch_size=10)
    for i,j in data:
        show(i[0])
        print('this is a dog!') if j[0]==1 else print('this is a cat!')
        break