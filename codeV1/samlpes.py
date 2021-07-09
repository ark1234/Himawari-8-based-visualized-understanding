import os
import numpy as np
import random
val_rate=0.2
train_csv='train.npy'
val_csv='val.npy'
data_list=os.listdir('data/mask')
data_len = len(data_list)
val_len = int(data_len * val_rate)
data_idx = random.sample(range(data_len), data_len)
val_idx = [data_list[i] for i in data_idx[:val_len]]
train_idx = [data_list[i] for i in data_idx[val_len:]]
train_idx=np.array(train_idx)
val_idx=np.array(val_idx)
np.save(train_csv,train_idx)
np.save(val_csv,val_idx)