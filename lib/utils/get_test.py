import random
import pandas as pd
import numpy as np
import keras
import os
import os.path
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import load_model
from lib.utils.misc import *
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

x_path='./train_val/train_val'
x_file=os.listdir(x_path)
x_test_path='./test/test'
model=load_model("model.h5")



def get_testdataset():
    x_return=[]
    x_name=pd.read_csv("sampleSubmission.csv") ['name']
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.84+x_voxel*0.16
        x_return.append(x_temp[34:66,34:66,34:66])

    x_test = np.array(x_return)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 32, 1)
    x_test = x_test.astype('float32') / 255
    score=get_score(model,x_test)

    return score




