import os,shutil

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

filespath='/home/zhan/下载/yellowcolorV2/yellowcolorV2'
names1list=os.listdir(filespath)
for name1 in names1list:
    path=filespath+'/'+name1
    names2list=os.listdir(path)
    for name2 in names2list:
        mymovefile(path + '/' + name2,filespath + '/' + name2)