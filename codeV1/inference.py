# coding=utf-8
import keras
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from read_data import *
from sklearn.metrics import confusion_matrix
from model import unet
PATH='data/test/image'
outpath='data/new_test'
ROOT_PRED='data/val'
model=unet()
model.load_weights('check_points/weights-improvement-10.hdf5')
nameslist=os.listdir(PATH)
matrixs=np.zeros([2,2],np.float32)
for i,name in enumerate(nameslist):
    print(name)
    nib_data = nib.load(PATH + '/' + name)
    imgs=nib_data.get_data()
    imgs_512 = cv2.resize(imgs, (img_w, img_h))
    preds=[]


    for j in range(imgs.shape[-1]):
        img = normalization(imgs_512[:, :, j])

        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)

        pred = model.predict(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred=np.squeeze(pred,0)
        pred=pred.astype(np.uint8)
        pred=np.rot90(pred)
        pred=cv2.resize(pred,(imgs.shape[1],imgs.shape[0]))
        preds.append(pred)
    preds=np.array(preds)
    preds=np.transpose(preds,[1,2,0])
    preds = nib.Nifti1Image(preds, nib_data.affine)
    nib.save(preds, outpath+'/mask_'+name)








