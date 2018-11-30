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

def printConCon(i,tp,fp,fn,tn):
    acc = (tp+tn)/(tp+fp+fn+tn)
    rec = tp/(tp+fn)
    pre = tp/(tp+fp)
    spe = tn/(tn+fp)
    fal = fp/(fp+tp)
    print('tp: ',tp," fp: ",fp," fn: ",fn," tn: ",tn)
    print('Accuracy: ', acc)
    print('Recall: ', rec)
    print('Precision: ', pre)
    print('Specificity: ', spe)
    print('Fall-out: ', fal)
    out = constraints[i]+","+str(ks)+","+str(fil)+","+str(no_epochs)+","+str(lstms)+","+str(tp)+","+str(fp)+","+str(fn)+","+str(tn)
    output.append(out)

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
    tp = np.zeros(no_con)
    fp = np.zeros(no_con)
    fn = np.zeros(no_con)
    tn = np.zeros(no_con)
    thres = 0.25
    print_size = int(len(test_y)/20)
    
    print('Generating predictions: ')
    
    for tr in range(0,len(test_y)):  
        for t, trace in enumerate(test_y[np.newaxis, tr, ::, ::, ::]):
            if tr%print_size==0:
                print("Trace: ", tr)
            prediction = model.predict(test_x[np.newaxis, tr, ::, ::, ::])
            for w, window in enumerate(trace):
                for a, act in enumerate(window):
                    for a2, act2 in enumerate(act):
                        for c, con in enumerate(act2):
                            x_act = prediction[0,w,a,a2]
                            if act2[c] > thres and x_act[c] > thres:
                                tp[c] += 1
                            if act2[c] > thres and x_act[c] < thres:
                                fp[c] += 1
                            if act2[c] < thres and x_act[c] > thres:
                                fn[c] += 1
                            if act2[c] < thres and x_act[c] < thres:
                                tn[c] += 1                                      
    for i in range(0,no_con):
        print('\nConstraint: ',constraints[i])
        printConCon(i,tp[i],fp[i],fn[i],tn[i])
    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)
    tn = np.sum(tn)
    return tp, fp, fn, tn

##############
### Main code 
dataset = 'bpi17_5'
limit = 40000
cutoff = 1
filt = 8
ks = 4
no_lstms = 1    
no_epochs = 15
   

output = []
output.append('element,ks,nofilt,noepoch,nolstsms,tp,fp,fn,tn')

print("Loading data")
data_X, data_y, no_traces, no_win, no_act, no_con = prepareDataConv()
print(dataset+ " loaded")

tp, fp, fn, tn = check_results()
acc = (tp+tn)/(tp+fp+fn+tn)
rec = tp/(tp+fn)
pre = tp/(tp+fp)
spe = tn/(tn+fp)
fal = fp/(fp+tp)

print('\nTotal: ')
print('Accuracy: ', acc)
print('Recall: ', rec)
print('Precision: ', pre)
print('Specificity: ', spe)
print('Fall-out: ', fal)
out = 'total,'+str(ks)+","+str(filt)+","+str(no_epochs)+","+str(no_lstms)+","+str(tp)+","+str(fp)+","+str(fn)+","+str(tn)
output.append(out)
    
outfile = open('output_co'+str(cutoff)+"_"+dataset+".csv", 'w')
for line in output:
    outfile.write(line+"\n")
outfile.flush()
outfile.close()