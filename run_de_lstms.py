import numpy as np
from keras import backend as K
from train_de_lstms import prepareDataConv, runModel, check_results

np.random.seed(7)

# Specify the dataset
dataset = 'bpi12_5n'
file_prefix = 'output_delstms_'


# Limits the number of traces used
limit = 1000

print("Loading data")
data_X, data_y, no_traces, no_win, no_act, no_con = prepareDataConv(dataset, limit)
vector_size = no_act * no_act * no_con
print(dataset + " loaded")

outfile = open(file_prefix + dataset + ".csv", 'a')
outfile.write(
    'element,no_ep,bs,opt,ld,no_lstms,ld_lstm,act_reg,kern_reg,time,no_param,ap,auc,recall,precision,fscore\n')
outfile.flush()
outfile.close()


############
# Parameters
############

# Number of epochs used
no_epochs = 10

# Activity and kernel regularisation
act_reg = 0
kern_reg = 0

# Number of LSTM layers
no_lstms = 2

# Dimensionality of encoder/decoder
ld = 64

print('Act. reg.: ', act_reg)
print('Kern. reg.: ', kern_reg)
print('Number of LSTM layers: ', no_lstms)
print('Dim. encoding.: ', ld)

# Training-test split
val_split = 0.8

# Batch size
bs = 20


###############
# Running model
###############

params = [no_epochs, bs, True, val_split, 1, ld, 0, no_lstms, act_reg, kern_reg]

# model string
model_name = 'model_' + dataset + '_ne' + str(no_epochs) + '_ar' + str(act_reg) + '_kr' + str(
    kern_reg)
model_name += '_ld' + str(ld) + '_ldls' + str(0) + '_o' + str(1)

# Run and store model
model, test_X, test_y, no_param, time = runModel(data_X, data_y, no_traces, no_win, no_act,
                                                 no_con, params)

model_json = model.to_json()
with open('./models/' + model_name + ".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('./models/' + model_name + ".h5")
print("Saved model to disk")
print('Name: ' + model_name)


##############
# Save results
##############
ap, auc, rec, prec, fs, coninfo = check_results(test_X, test_y, model_name, no_win, no_act, no_con)

out = 'total,' + str(no_epochs) + ',' + str(bs) + ',' + str(1) + ',' + str(ld) + ',' + str(
    no_lstms) + ',' + str(0) + ',' + str(act_reg) + ',' + str(kern_reg) + ',' + str(
    time) + ',' + str(no_param) + ',' + str(ap) + ',' + str(auc) + "," + str(rec) + "," + str(
    prec) + "," + str(fs)

K.clear_session()

outfile = open(file_prefix + dataset + ".csv", 'a')
outfile.write(out + "\n")
outfile.flush()
outfile.close()