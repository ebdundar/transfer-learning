# transfer-learning
Transfer learning example by using CIFAR-10 dataset and Keras Framework. 

The aim of this project is to use parameters which are previously learned for training new classes. At first, 2 classes are trained for 20 epochs. Then, convolutional layers are frozen since they learned how to extract simple features. Furthermore, images belonging to 2 new classes are trained for 10 epochs since we do not train conv layers. Finally, final layers of two different models are merged. So, we have a model with 4 targets. This model is trained for 3 epochs.
Note: Number of epochs are determined by empirically. So, you are free to change.

## Files
* train.py
It is used when the first 2 classed are trained.
* transfer.py
In this file, some trainable parameters are frozen so that previously learned weights are utilized for training new classes.
* merged_model.py
Two different models are combined together in this file.

## How should these files be executed?

The order is the following: "train.py", "transfer.py", and "merged_model.py".
