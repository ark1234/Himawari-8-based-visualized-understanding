# coding=utf-8
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import random
import math
import os
# import nibabel as nib


img_w = int(int(545/32)*32)
img_h = int(int(750/32)*32)
print(img_h,img_w)


def normalization(img):
    min=np.min(img)
    max=np.max(img)
    img=(img-min)/(max-min)
    return img

def get_train_val(filepath_train,filepath_val):

    train_set=np.load(filepath_train)
    val_set = np.load(filepath_val)

    return train_set, val_set

def generateTrainData(batch_size,root1,root2,name):
    #print 'generateData...'
    np.random.shuffle(name)
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(name))):
            url = name[i]
            img_1=cv2.imread(root1 + '/' + url[7:-4]+'_1.tif',0)
            img_2=cv2.imread(root1 + '/' + url[7:-4]+'_2.tif',0)
            img_3=cv2.imread(root1 + '/' + url[7:-4]+'_3.tif',0)
            img_4=cv2.imread(root1 + '/' + url[7:-4]+'_4.tif',0)
            img=np.stack([img_1,img_2,img_3,img_4],-1)

            img=cv2.resize(img,(img_w,img_h))/255.0
            #img=normalization(img)


            label=cv2.imread(root2+'/'+url)[:,:,0]
            label=cv2.resize(label,(img_w,img_h))
            label[label>50]=1
            label[label<=50]=0
            label=np.expand_dims(label,-1)
            batch += 1
            train_data.append(img)
            train_label.append(label)
            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data,train_label)
                train_data = []
                train_label = []
                batch = 0



