import cv2
import numpy
import os
namelist=os.listdir('data/mask')
print(len(namelist))
for name in namelist:
    if( not os.path.exists('data/rsdata' + '/' + name[7:-4]+'_1.tif')):
        print(name)


