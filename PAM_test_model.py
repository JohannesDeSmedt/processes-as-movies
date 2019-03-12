import numpy as np
import os
from os import listdir
from os.path import isfile, join, splitext

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
import time
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from numpy import array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# fix random seed for reproducibility
np.random.seed(7)

constraints = {}
constraints[0] = "Precedence"
constraints[1] = "AlternatePrecedence"
constraints[2] = "ChainPrecedence"
constraints[3] = "Response"
constraints[4] = "AlternateResponse"
constraints[5] = "ChainResponse"
constraints[6] = "Absence"
constraints[7] = "Exactly"
constraints[8] = "Exactly2"
constraints[9] = "Existence3"
constraints[10] = "Init"
constraints[11] = "Last"
constraints[12] = "NotSuccession"
constraints[13] = "CoExistence"

def prepareDataConv():
    file = open(dataset+'.txt', 'r')
    
    line1 = file.readline()
    no_con = int(line1)
    no_act = int(file.readline())
    
    no_traces = 0
    no_win = 0
    no_feat = 0
    
    traces_X = []
    traces_y = []
    p = 0
    window_list = []
    for line in file:
        if line == '\n':
            if no_traces%1000==0:
                print('new trace added: '+str(no_traces))
            traces_X.append(array(window_list[:-cutoff]))
            traces_y.append(array(window_list[cutoff:]))
            no_traces += 1
            window_list = []
            
            if no_traces == limit:
                break
            no_win = p
            p = 0 
        else:
            no = line.split(',')
            no_feat = len(no)
            for i in range(0,no_feat):
                no[i] = int(no[i])
            
            act1_list = []
            for act1 in range(0,no_act):
                act2con_list = []
                for act2 in range(0,no_act):
                    index = act1 * no_con * no_act + act2 * no_con
                    act2con_list.append(array(no[index:index+no_con]))
                act2_arr = array(act2con_list)
                act1_list.append(act2_arr)
            act1_arr = array(act1_list)
            window_list.append(act1_arr)
            p+=1
    traces_X = array(traces_X)
    traces_y = array(traces_y)
        
    print('#act: \t'+str(no_act))
    print('#constraints: \t'+str(no_con))
    print('#traces: \t'+str(no_traces))
    print('#windows: \t'+str(no_win))
    return traces_X, traces_y, no_traces, no_win-1, no_act, no_con

def check_results():
           
    train_x, test_x, train_y, test_y = train_test_split(data_X,data_y,test_size=0.2)
         
    model = "model_co"+str(cutoff)+"_"+dataset+"_ep"+str(no_epochs)+"_nl"+str(no_lstms)+"_fi"+str(filt)+"_ks"+str(ks)
    
    print('Opening model')
    json_file = open(model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print('Opening weights')
    loaded_model.load_weights(model+".h5")
    print("Loaded model from disk")
        
    model = loaded_model
#    print('Prediction: \n'+str(loaded_model.predict(test_x[np.newaxis, 2, ::, ::, ::])))

    print_size = int(len(test_y)/20)
    
    print('Generating predictions: ')    
    y_pred = np.zeros((len(test_y), no_win, no_act, no_act, no_con), dtype=np.float32)
    y_pred_i = np.zeros((len(test_y), no_win, no_act, no_act, no_con), dtype=np.int32)
    y_val = np.zeros((len(test_y), no_win, no_act, no_act, no_con), dtype=np.int32)
    
    sc = 0
    for tr in range(0,len(test_y)):  
        for t, trace in enumerate(test_y[np.newaxis, tr, ::, ::, ::]):
            if tr%print_size==0:
                print("Trace: ", tr,' #',sc,'/20')
                sc += 1
            prediction = model.predict(test_x[np.newaxis, tr, ::, ::, ::])
            for w, window in enumerate(trace):
                for a, act in enumerate(window):
                    for a2, act2 in enumerate(act):
                        for c, con in enumerate(act2):
                            pred = prediction[0,w,a,a2,c]
                            act = trace[w,a,a2,c]
                            y_val[tr,w,a,a2,c] = act
                            y_pred[tr,w,a,a2,c] = pred  

    y_pred = np.reshape(y_pred,(len(y_val)*no_win*no_act*no_act*no_con))
    y_val = np.reshape(y_val,(len(y_val)*no_win*no_act*no_act*no_con))   
    y_pred_i = np.reshape(y_pred_i,(len(y_pred_i)*no_win*no_act*no_act*no_con))
        
    # Finding threshold
    average_precision = average_precision_score(y_val, y_pred)
    precision, recall, pr_thres = precision_recall_curve(y_val,y_pred)
    best_score = 0
    best_thres = 0
    for thres, prec, rec in zip(pr_thres, precision, recall):
        if (2*(prec*rec)/(prec+rec)) > best_score:
            best_thres= thres
            best_score= 2*(prec*rec)/(prec+rec)
    print('Best threshold for F-score: ', best_thres)
    print('Average precision: ', average_precision)
    for i in range(0,len(y_pred)):
        if y_pred[i]>=best_thres:
            y_pred_i[i] = 1

    auc = roc_auc_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred_i)
    precision = precision_score(y_val, y_pred_i)
    fscore = 2*(precision*recall)/(precision+recall)
    print('AUC: ', auc)   
    print('Recall: ', recall)  
    print('Precision: ', precision)                   
    print('F-score: ', fscore)   

    return average_precision, auc, recall, precision, fscore


##############
### Main code 
dataset = 'bpi12_5'
limit = 100000
cutoff = 1
filt = 9
ks = 9
no_lstms = 0
no_epochs = 15


output = []
output.append('element,ks,nofilt,noepoch,nolstsms,avprec,auc,recall,precision,fscore')

print("Loading data")
data_X, data_y, no_traces, no_win, no_act, no_con = prepareDataConv()
print(dataset+ " loaded")

ap, auc, rec, prec, fs = check_results()
out = 'total,'+str(ks)+","+str(filt)+","+str(no_epochs)+","+str(no_lstms)+","+str(ap)+","+str(auc)+","+str(rec)+","+str(prec)+","+str(fs)
output.append(out)


outfile = open('output_co'+str(cutoff)+"_"+dataset+".csv", 'w')
for line in output:
    outfile.write(line+"\n")
outfile.flush()
outfile.close()