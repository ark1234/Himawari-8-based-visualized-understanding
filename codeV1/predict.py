# coding=utf-8
import keras
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from read_data import *
from sklearn.metrics import confusion_matrix
from model import unet
PATH='data/test/image'
root1='data/rsdata'
root2='data/mask'
ROOT_PRED='data/val'
img_h,img_w= 544,736
model=unet()
model.load_weights('check_points/weights-improvement-63.hdf5')
#nameslist=os.listdir(PATH)
nameslist=np.load('val.npy')
matrixs=np.zeros([2,2],np.float32)
for i,name in enumerate(nameslist):
    url=name
    print(name)
    img_1 = cv2.imread(root1 + '/' + url[7:-4] + '_1.tif', 0)
    img_2 = cv2.imread(root1 + '/' + url[7:-4] + '_2.tif', 0)
    img_3 = cv2.imread(root1 + '/' + url[7:-4] + '_3.tif', 0)
    img_4 = cv2.imread(root1 + '/' + url[7:-4] + '_4.tif', 0)
    img = np.stack([img_1, img_2, img_3, img_4], -1)
    img = cv2.resize(img, (img_w, img_h)) / 255.0

    img = np.expand_dims(img,0)
    # img=normalization(img)

    label = cv2.imread(root2 + '/' + url)[:, :, 0]
    print(label.shape)
    label = cv2.resize(label, (img_w, img_h))

    label[label <= 20] = 0
    label[label > 20] = 1
    pred=model.predict(img)
    print(pred.shape)
    pred[pred>0.5]=1
    pred[pred<=0.5]=0

    output=pred.copy()
    pred[pred == 1] = 255
    pred=np.squeeze(pred,-1)
    pred=np.squeeze(pred,0)
    target=label
    target[target > 0] = 255
    if not os.path.exists(ROOT_PRED):
        os.makedirs(ROOT_PRED)
    '''cv2.imwrite(ROOT_PRED + '/' + name[:-4] + '1.png', img_1)
    cv2.imwrite(ROOT_PRED + '/' + name[:-4] + '2.png', img_2)
    cv2.imwrite(ROOT_PRED + '/' + name[:-4] + '3.png', img_3)
    cv2.imwrite(ROOT_PRED + '/' + name[:-4] + '4.png', img_4)'''

    cv2.imwrite(ROOT_PRED + '/' + name[:-4] + 'pred.png', pred)
    cv2.imwrite(ROOT_PRED + '/' + name[:-4] + 'gt.png', target)
    target[target > 0] = 1
    target = np.reshape(target, [-1])
    output = np.reshape(output, [-1])
    target = target.astype(np.int8)
    output = output.astype(np.int8)
    labels = list(set(np.concatenate((target, output), axis=0)))
    print(labels)
    matrixs_temp = np.zeros([2, 2], np.float32)

    if (labels == [0]):
        matrixs[0, 0] += confusion_matrix(target, output)[0, 0]
        matrixs_temp[0, 0] = confusion_matrix(target, output)[0, 0]
    elif (labels == [1]):
        matrixs[1, 1] += confusion_matrix(target, output)[0, 0]
        matrixs_temp[1, 1] = confusion_matrix(target, output)[0, 0]
    else:
        matrixs += confusion_matrix(target, output)
        matrixs_temp = confusion_matrix(target, output)
    print(matrixs_temp)

confusion_matrixs = pd.DataFrame(matrixs)
confusion_matrixs.to_csv('confusion_matrix.csv', header=None, index=None)



