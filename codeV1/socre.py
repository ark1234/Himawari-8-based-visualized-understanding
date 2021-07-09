import pandas as pd

table = pd.read_table('confusion_matrix.csv',header=None, sep=',')
matrixs=table.values
a,b,c,d=matrixs[0][0],matrixs[0][1],matrixs[1][0],matrixs[1][1]
accuracy=(a+d)/(a+b+c+d)
recall = d/(d + c)
precision = d/(d + b)
F1=2*d/(a+b+c+d+d-a)
iou=d/(c+b+d)
print(matrixs)
print('accuracy:',accuracy)
print('recall:',recall)
print('precision:',precision)
print('F1:',F1)
print('iou:',iou)