# processes-as-movies

This project provides both a Java-based feature generation procedure for generating Declare activity tensors from process execution logs, and Python code for building convolutional recurrent neural networks based on these tensors. 

## Feature generation
The feature generation procedure uses iBCM (https://feb.kuleuven.be/public/u0092789/) to find constraints present in execution traces, and stores them in a .txt file.
The d2v.jar file takes two arguments:
-- -w for the number of windows
-- -l for the event log (which should be XES-based, don't include the .xes extension)

E.g. java -jar d2v.jar -w 10 -l BPI_Challenge_2012

## Convolutional recurrent neural network
The Python files can be used for training a model, and the subsequent testing. There are a number of parameters, which can be set in the code itself:
-- filt: number of filters to be used for max pooling
-- ks: kernel size
-- no_lstms: additional layers of CONVLSTMs
-- no_epochs: the number of epochs to traing over the network
-- cutoff: set to 1, it reads full traces, set to 2, it leaves a gap between input and output window
