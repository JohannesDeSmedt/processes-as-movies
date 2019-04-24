import numpy as np

import keras
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from numpy import array
from sklearn.model_selection import train_test_split

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
                print('new trace '+str(no_traces))
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
#    traces_y = np.reshape(traces_y, (len(traces_y), 1, 24,24,14))
            
    print('#act: \t'+str(no_act))
    print('#constraints: \t'+str(no_con))
    print('#traces: \t'+str(no_traces))
    print('#windows: \t'+str(no_win))
    print(np.shape(traces_X))
    print(np.shape(traces_y))
    return traces_X, traces_y, no_traces, no_win-1, no_act, no_con

def runLSTM(no_epochs, data_X, data_y, no_traces, no_win, no_act, no_con):
               
    train_x, test_x, train_y, test_y = train_test_split(data_X,data_y,test_size=0.2)
     
    time_callback = TimeHistory()
    model = Sequential()
    model.add(ConvLSTM2D(filters=filt, kernel_size=(ks, ks),
                       input_shape=(None, no_act, no_act, no_con),
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    for i in range(0,no_lstms):
        model.add(ConvLSTM2D(filters=filt, kernel_size=(ks, ks),
                       padding='same', return_sequences=True))
        model.add(BatchNormalization())
   
    
    model.add(Conv3D(filters=no_con, kernel_size=(ks, ks, ks),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=50,
        epochs=no_epochs, validation_split=0.1, callbacks=[time_callback])
        
    # Final evaluation of the model
    scores = model.evaluate(test_x, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    model_name = "model_co"+str(cutoff)+"_"+dataset+"_ep"+str(no_epochs)+"_nl"+str(no_lstms)+"_fi"+str(filt)+"_ks"+str(ks)
    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name+".h5")
    print("Saved model to disk")
    print('Name: '+model_name)
        

##############
### Main code
dataset = 'bpi12_5'

# Limits the number of traces used
limit = 1000
# Number of windows to skip
cutoff = 1
# Filter size
filt = 8
# Kernel size
ks = 4
# Number of extra LSTM layers
no_lstms = 1    
no_epochs = 15

output = []
output.append('element,ks,nofilt,noepoch,nolstsms,tp,fp,fn,tn')

print("Loading data")
data_X, data_y, no_traces, no_win, no_act, no_con = prepareDataConv()
print(dataset+ " loaded")

print("\n\n\n########### New iteration ############")
print('#Kernel size: ', ks)
print('#Filters: ', filt)
print('#Epochs: ', no_epochs)
print('#LSTM layers: ', no_lstms)
print('Cutoff: ', cutoff)                  
      
runLSTM(no_epochs, data_X, data_y, no_traces, no_win, no_act, no_con)