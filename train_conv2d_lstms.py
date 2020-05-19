import numpy as np

import keras
import time
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from numpy import array
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split

from keras.models import model_from_json
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve


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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, no_win, no_act, no_con, shuffle):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.no_win = no_win
        self.no_act = no_act
        self.no_con = no_con        
        self.list_IDs = list_IDs       
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.no_win, self.no_act, self.no_act, self.no_con))
        y = np.empty((self.batch_size, 1, self.no_act, self.no_act, self.no_con))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data_conv2d/traces_'+str(self.no_win)+'_x_' + str(ID) + '.npy')
            # Store class
            y[i,] = np.load('data_conv2d/traces_'+str(self.no_win)+'_y_' + str(ID) + '.npy')
        
#        vector_size = self.no_act * self.no_act * self.no_con
        y = np.reshape(y, (len(y), 1, self.no_act, self.no_act, self.no_con))
        return X, y


def prepareDataConv(dataset, limit):
    file = open(dataset+'.txt', 'r')
    
    line1 = file.readline()
    no_con = int(line1)
    no_act = int(file.readline())
    
    no_traces = 0
    no_win = 0
    
    traces_X = []
    traces_y = []
    p = 0
    window_list = []
    for line in file:
        if line == '\n':
            if no_traces%1000==0:
                print('new trace '+str(no_traces))
            traces_X.append(array(window_list[:-1]))
            traces_y.append(array(window_list[-1]))
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
    print(np.shape(traces_y))
    traces_y = array(traces_y)
    traces_y = np.reshape(traces_y, (len(traces_y), 1, no_act,no_act,no_con))

    for i in range(0,len(traces_X)):        
        np.save('./data_conv2d/traces_'+str(no_win-1)+'_x_'+str(i), traces_X[i])
        np.save('./data_conv2d/traces_'+str(no_win-1)+'_y_'+str(i), traces_y[i])
            
    print('#act: \t'+str(no_act))
    print('#constraints: \t'+str(no_con))
    print('#traces: \t'+str(no_traces))
    print('#windows: \t'+str(no_win))
    print(np.shape(traces_X))
    print(np.shape(traces_y))
    return traces_X, traces_y, no_traces, no_win-1, no_act, no_con


def createCONV2D_LSTM(no_win, no_act, no_con, params):
    ks = params[0]
    no_lstms = params[1]
    filt = params[2]
    act_reg = params[3]
    kern_reg = params[4]    
        
    model = Sequential()
    model.add(ConvLSTM2D(filters=filt, kernel_size=(ks, ks),
                       input_shape=(None, no_act, no_act, no_con),
                       padding='same', return_sequences=True,kernel_regularizer=l2(kern_reg),
                         activity_regularizer=l2(act_reg)))
    model.add(BatchNormalization())
    
    for i in range(0, no_lstms):
        model.add(ConvLSTM2D(filters=filt, kernel_size=(ks, ks),
                       padding='same', return_sequences=True,kernel_regularizer=l2(kern_reg),
                             activity_regularizer=l2(act_reg)))
        model.add(BatchNormalization())
       
    model.add(Conv3D(filters=no_con, kernel_size=(ks, ks, ks),
                   activation='sigmoid',
                   padding='same', data_format='channels_last',kernel_regularizer=l2(kern_reg),
                     activity_regularizer=l2(act_reg)))
    
    return model


def runModel(data_X, data_y, no_traces, no_win, no_act, no_con, params):
    time_callback = TimeHistory()
    
    no_epochs = params[0]
    bs = params[1]
    val_split = params[3]
    opt = params[4]
    
    model = createCONV2D_LSTM(no_win, no_act, no_con, params[5:])
     
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y,test_size=(1-val_split))
    print(model.summary())
            
  
    partition = {}
    full_range = list(range(0, no_traces))
    np.random.shuffle(full_range)
    
    split_index = int(no_traces*val_split)
    split_val_index = int(split_index*val_split)
    partition['train'] = full_range[0:split_val_index]
    partition['validation'] = full_range[split_val_index:split_index]
    partition['test'] = full_range[split_index:no_traces]
    test_X = data_X[partition['test']]
    test_y = data_y[partition['test']]
    
    train_gen = DataGenerator(partition['train'], bs, no_win, no_act, no_con, False)
    val_gen = DataGenerator(partition['validation'], bs, no_win, no_act, no_con, False)
    test_gen = DataGenerator(partition['test'], bs, no_win, no_act, no_con, False)
    
    if opt == 1:
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])     
    else:
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])   
       

    model.fit_generator(generator = train_gen, validation_data = val_gen,
                        epochs=no_epochs, callbacks=[time_callback], verbose=1)   
    scores = model.evaluate_generator(test_gen)

    print('Epoch length (s): ', time_callback.times)
    time = np.mean(time_callback.times)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    return model, test_X, test_y, model.count_params(), time


def check_results(test_X, test_y, model_name, no_win, vector_size, no_act, no_con):               
    print('Opening model')
    json_file = open('./models/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    print('Opening weights')
    model.load_weights('./models/'+model_name+".h5")
    print("Loaded model from disk")
        
    print_size = int(len(test_y)/20)
    
    print('X shape: ', test_X.shape)
    print('y shape:' , test_y.shape)
    
    print('Generating predictions: ')    
    y_pred = np.zeros((len(test_y), 1, no_act, no_act, no_con), dtype=np.float32)
    y_pred_i = np.zeros((len(test_y), 1, no_act, no_act, no_con), dtype=np.int32)
    y_val = np.zeros((len(test_y), 1, no_act, no_act, no_con), dtype=np.int32)

    y_pred_dict = {}
    y_val_dict = {}
    for c in range(0,no_con):
        y_pred_dict[c] = np.zeros((len(test_y),no_act,no_act),dtype=np.float32)
        y_val_dict[c] = np.zeros((len(test_y),no_act,no_act),dtype=np.float32)
    
    sc = 0
    for tr in range(0,len(test_y)):  
        for t, trace in enumerate(test_y[np.newaxis, tr, ::, ::, ::]):
            if tr%print_size==0:
                print("Trace: ", tr,' #',sc,'/20')
                sc += 1
            prediction = model.predict(test_X[np.newaxis, tr, ::, ::, ::])
            if tr == 0:
                print('Shape text_X: ', np.shape(test_X[np.newaxis, tr]))
                print('Shape prediction: ', np.shape(prediction))
            for w, window in enumerate(trace):
                if tr==0:
                    print('W: ', w)
                for a, act in enumerate(window):
                    for a2, act2 in enumerate(act):
                        for c, con in enumerate(act2):
                            pred = prediction[0,no_win-1,a,a2,c]
                            act = trace[w,a,a2,c]

                            y_val[tr,w,a,a2,c] = act
                            y_pred[tr,w,a,a2,c] = pred  
                                                 
                            y_pred_dict[c][tr,a,a2] = pred
                            y_val_dict[c][tr,a,a2] = act

    y_pred = np.reshape(y_pred,(len(y_val)*1*vector_size))
    y_val = np.reshape(y_val,(len(y_val)*1*vector_size))   
    y_pred_i = np.reshape(y_pred_i,(len(y_pred_i)*1*vector_size))
        
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
    
    coninfo = []
    for c in range(0,no_con):
        print('\nConstraint: ', constraints[c])
        flat_pred = np.reshape(y_pred_dict[c],(len(y_pred_dict[c])*no_act*no_act))
        flat_val = np.reshape(y_val_dict[c],(len(y_val_dict[c])*no_act*no_act))
        
        print('#occurrences: ',np.sum(flat_val))
        auc = ap = 0
        if np.sum(flat_val) > 0:    
            auc = roc_auc_score(flat_val,flat_pred)
            ap = average_precision_score(flat_val,flat_pred)
            print('AUC: ',auc)
            print('AP: ',ap)
        coninfo.append(constraints[c]+','+str(np.sum(flat_val))+','+str(auc)+','+str(ap)+'\n')

    return average_precision, auc, recall, precision, fscore, coninfo