# processes-as-movies (PAM)

This project provides both a Java-based feature generation procedure for generating Declare activity tensors from process execution logs, and Python code for building convolutional recurrent neural networks based over these tensors. 

## Feature generation
The feature generation procedure uses [iBCM](https://feb.kuleuven.be/public/u0092789/) to find constraints present in execution traces, and stores them in a .txt file.
The d2v.jar file takes two arguments:
<ul>
<li> -w for the number of windows
<li> -l for the event log (which should be XES-based (http://www.xes-standard.org/openxes/start), don't include the .xes extension)
</ul>
E.g. ´´

Inline `java -jar d2v.jar -w 10 -l BPI_Challenge_2012` has `back-ticks around` it.

## Convolutional recurrent neural network
The Python files can be used for training a model, and the subsequent testing. There are a number of parameters, which can be set in the code itself:
<ul>
<li> filt: number of filters to be used for max pooling
<li> ks: kernel size
<li> no_lstms: additional layers of CONVLSTMs
<li> no_epochs: the number of epochs to traing over the network
<li> cutoff: set to 1, it reads full traces, set to 2, it leaves a gap between input and output window
</ul>

PAM makes use of [Keras](https://keras.io/) and [scikit-learn](https://scikit-learn.org/stable/).
