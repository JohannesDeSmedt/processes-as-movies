# processes-as-movies (PAM)

This project provides both a Java-based feature generation procedure for generating Declare activity tensors from process execution logs, and Python code for building convolutional recurrent neural networks based over these tensors. 

## Feature generation
The feature generation procedure uses [iBCM](https://feb.kuleuven.be/public/u0092789/) to find constraints present in execution traces, and stores them in a .txt file.
The d2v.jar file takes two arguments:
* -w for the number of windows
* -l for the event log (which should be XES-based (http://www.xes-standard.org/openxes/start), don't include the .xes extension)

For example: `java -jar d2v.jar -w 10 -l BPI_Challenge_2012` 

The logs used to create the datasets in /datasets are:
* [BPI Challenge 2017](https://data.4tu.nl/repository/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)
* [BPI Challenge 2012](https://data.4tu.nl/repository/uuid:3926db30-f712-4394-aebc-75976070e91f)
* [BPI Challenge 2013 - Incidents](https://data.4tu.nl/repository/uuid:500573e6-accc-4b0c-9576-aa5468b10cee)

## Convolutional recurrent neural network
The Python files can be used for training a model, and the subsequent testing. There are a number of parameters, which can be set in the code itself:
* `filt`: number of filters to be used for max pooling
* `ks`: kernel size
* `no_lstms`: additional layers of CONVLSTMs
* `no_epochs`: the number of epochs to traing over the network
* `cutoff`: set to 1, it reads full traces, set to 2, it leaves a gap between input and output window

PAM makes use of [Keras](https://keras.io/) and [scikit-learn](https://scikit-learn.org/stable/).

## Parameter results
Y-axes: average precision

X-axes CONVLSTM: kernel size

### CONVLSTMs - Fixed number of windows:

BPI 12, 2 windows:
![BPI 12, 2 windows](/results/CONVbpi122.png)
BPI 12, 5 windows:
![BPI 12, 5 windows](/results/CONVbpi125.png)
BPI 12, 10 windows:
![BPI 12, 10 windows](/results/CONVbpi1210.png)
BPI 17, 2 windows:
![BPI 17, 2 windows](/results/CONVbpi172.png)
BPI 17, 5 windows:
![BPI 17, 5 windows](/results/CONVbpi175.png)
BPI 17, 10 windows:
![BPI 17, 10 windows](/results/CONVbpi1710.png)

### Encoder-decoder LSTMs - Fixed number of windows:
BPI 12, 2 windows:
![BPI 12, 2 windows](/results/DEbpi122.png)
BPI 12, 5 windows:
![BPI 12, 5 windows](/results/DEbpi125.png)
BPI 12, 10 windows:
![BPI 12, 10 windows](/results/DEbpi1210.png)
BPI 17, 2 windows:
![BPI 17, 2 windows](/results/DEbpi172.png)
BPI 17, 5 windows:
![BPI 17, 5 windows](/results/DEbpi175.png)
BPI 17, 10 windows:
![BPI 17, 10 windows](/results/DEbpi1710.png)

### CONVLSTMs - Window size 2:
BPI 12, 6-10 windows:
![BPI 12, 2 windows](/results/CONVbpi12210u20.png)
BPI 12, 11-15 windows:
![BPI 12, 2 windows](/results/CONVbpi12220u30.png)
BPI 12, 16-20 windows:
![BPI 12, 2 windows](/results/CONVbpi12230u40.png)
BPI 12, 21-25 windows:
![BPI 12, 2 windows](/results/CONVbpi12240u50.png)
BPI 12, 26-30 windows:
![BPI 12, 2 windows](/results/CONVbpi12250u60.png)

BPI 17, 6-10 windows:
![BPI 17, 2 windows](/results/CONVbpi17210u20.png)
BPI 17, 11-15 windows:
![BPI 17, 2 windows](/results/CONVbpi17220u30.png)
BPI 17, 16-20 windows:
![BPI 17, 2 windows](/results/CONVbpi17230u40.png)
BPI 17, 21-25 windows:
![BPI 17, 2 windows](/results/CONVbpi17240u50.png)
BPI 17, 26-30 windows:
![BPI 17, 2 windows](/results/CONVbpi17250u60.png)

### CONVLSTMs - Window size 5:
BPI 12, 3-4 windows:
![BPI 12, 2 windows](/results/CONVbpi12510u20.png)
BPI 12, 5-6 windows:
![BPI 12, 2 windows](/results/CONVbpi12520u30.png)
BPI 12, 7-8 windows:
![BPI 12, 2 windows](/results/CONVbpi12530u40.png)
BPI 12, 9-10 windows:
![BPI 12, 2 windows](/results/CONVbpi12540u50.png)
BPI 12, 11-12 windows:
![BPI 12, 2 windows](/results/CONVbpi12550u60.png)

BPI 17, 3-4 windows:
![BPI 17, 2 windows](/results/CONVbpi17510u20.png)
BPI 17, 5-6 windows:
![BPI 17, 2 windows](/results/CONVbpi17520u30.png)
BPI 17, 7-8 windows:
![BPI 17, 2 windows](/results/CONVbpi17530u40.png)
BPI 17, 9-10 windows:
![BPI 17, 2 windows](/results/CONVbpi17540u50.png)
BPI 17, 11-12 windows:
![BPI 17, 2 windows](/results/CONVbpi17550u60.png)

### CONVLSTMs - Window size 10:
BPI 12, 2 windows:
![BPI 12, 2 windows](/results/CONVbpi121010u20.png)
BPI 12, 3 windows:
![BPI 12, 2 windows](/results/CONVbpi121020u30.png)
BPI 12, 4 windows:
![BPI 12, 2 windows](/results/CONVbpi121030u40.png)
BPI 12, 5 windows:
![BPI 12, 2 windows](/results/CONVbpi121040u50.png)
BPI 12, 6 windows:
![BPI 12, 2 windows](/results/CONVbpi121050u60.png)

BPI 17, 2 windows:
![BPI 17, 2 windows](/results/CONVbpi171010u20.png)
BPI 17, 3 windows:
![BPI 17, 2 windows](/results/CONVbpi171020u30.png)
BPI 17, 4 windows:
![BPI 17, 2 windows](/results/CONVbpi171030u40.png)
BPI 17, 5 windows:
![BPI 17, 2 windows](/results/CONVbpi171040u50.png)
BPI 17, 6 windows:
![BPI 17, 2 windows](/results/CONVbpi171050u60.png)

### Encoder-decoder LSTMs - Fixed window length:
BPI 12:
![BPI 12](/results/DEbpi12.png)
BPI 17:
![BPI 12](/results/DEbpi17.png)
