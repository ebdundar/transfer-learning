from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.optimizers import SGD, Adam

import scipy.io
import numpy as np

np.random.seed(1337)  # for reproducibility

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

batch_size = 32
nb_classes = 4
nb_epoch = 3
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

trainmat = scipy.io.loadmat('cifar-0136.mat', verify_compressed_data_integrity=False)


X_train = np.array(trainmat['newData'])
y_train = np.array(trainmat['newLabels'])
X_test = np.array(trainmat['newData_test'])
y_test = np.array(trainmat['newLabels_test'])

X_train = X_train.reshape(20000, 32, 32, 3)
X_test = X_test.reshape(4000, 32, 32, 3)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


feature_layers = [
    Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]),
    Activation('relu'),
    Convolution2D(32, 3, 3),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Convolution2D(64, 3, 3, border_mode='same'),
    Activation('relu'),
    Convolution2D(64, 3, 3),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
]
classification_layers = [
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(nb_classes),
    Activation('softmax')
]

model = Sequential(feature_layers + classification_layers)

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#load model1
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model1 from disk")

#load model2
json_file = open('model_transfer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights into new model
model2.load_weights("model_transfer.h5")
print("Loaded model2 from disk")


weights = model.layers[16].get_weights()
weights2 = model2.layers[16].get_weights()

new_vars = np.float32(np.random.rand(512,4)) / 10
new_vars[:,0] = weights[0][:,0]
new_vars[:,1] = weights[0][:,1]

new_vars[:,2] = weights2[0][:,0]
new_vars[:,3] = weights2[0][:,1]

pop_layer(model);
pop_layer(model);

model.add(Dense(4, activation='softmax'))

weights[0] = new_vars
# get bias weights
a = np.array([weights[1][0], weights[1][1], weights2[1][0], weights2[1][1]])
weights[1] = a;

model.layers[16].set_weights(weights)

# trainable parameters are frozen
for l in feature_layers:
    l.trainable = False


# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.2,
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))

# save model
model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# save weights
model.save_weights("model_final.h5")
print("Saved model to disk")

score = model.evaluate(X_test, Y_test, verbose=1)
print (score)

preds = model.predict(X_test, batch_size=32)


count = [0,0,0]
Y_pred = []
for row in preds:
    count[list(row).index(max(row))] +=1
    Y_pred.append(list(row).index(max(row)))
print(count)

YY_test = []
for yy in Y_test:
    YY_test.append(list(yy).index(max(yy)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_pred, YY_test))

