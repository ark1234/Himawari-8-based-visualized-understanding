# coding=utf-8
import keras
from keras.callbacks import ModelCheckpoint
from read_data import *
from model import unet
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
path_checkpoints='check_points'
IMGPATH='/Volumes/Seagate Backup Plus Drive/output01'
MASKPATH='/Volumes/Seagate Backup Plus Drive/yellowcolor01'
EPOCHS = 200
BS = 6
model = unet()
#model.load_weights('check_points/weights-improvement.hdf5')

fileroot = path_checkpoints
filepath = os.path.join(fileroot, 'weights-improvement-{epoch:02d}.hdf5')
if (not os.path.exists(fileroot)):
    os.makedirs(fileroot)
modelcheck = ModelCheckpoint(filepath, monitor='val_acc', mode='max', verbose=1,save_weights_only=True)
tb_cb = keras.callbacks.TensorBoard(log_dir='log/exp2')
callable = [modelcheck, tb_cb]
train_set, val_set = get_train_val("train.npy",
                                   "val.npy")
train_numb = len(train_set)
valid_numb = len(val_set)
print("the number of train data is", train_numb)
print("the number of val data is", valid_numb)
#model.load_weights('check_points/weights-improvement-30.hdf5')
model.fit_generator(generator=generateTrainData(BS,IMGPATH,MASKPATH,train_set), steps_per_epoch=train_numb // BS,
                    validation_data=generateTrainData(BS,IMGPATH,MASKPATH, val_set), validation_steps=valid_numb // BS,
                    epochs=EPOCHS,
                    callbacks=callable, workers=1)
# plot the training loss and accuracy
# plot the training loss and accuracy