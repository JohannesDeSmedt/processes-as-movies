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

![Image description](/results/CONVbpi122.png)
