from skimage import io,transform
import glob
import os
import numpy as np

#数据集地址
path='E:\\data\\dataset\\man_woman\\train\\'

w=100
h=100
c=3

#读取图片
def read_img(path):
    cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)
    imgs = []
    labels = []
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s' % (im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)


def get_batch ():
    data,label=read_img(path)
    #打乱顺序
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]
    #将所有数据分为训练集和验证集
    ratio=0.8
    s=np.int(num_example*ratio)
    x_train=data[:s]
    y_train=label[:s]
    x_val=data[s:]
    y_val=label[s:]
    return x_train,y_train,x_val,y_val #ndarray

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

