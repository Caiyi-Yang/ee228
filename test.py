import pandas as pd     
import numpy as np
import keras
import os
import os.path
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from lib.utils.get_test import get_testdataset
import random

x_path='./train_val/train_val'
x_file=os.listdir(x_path)
x_test_path='./test/test'
model=load_model("model.h5")


score=get_testdataset()
csv = pd.read_csv("./sampleSubmission.csv")
csv.iloc[:, 1] = score[:, 1]
csv.columns = ['name', 'predicted']
csv.to_csv("./submission.csv",index=None)


