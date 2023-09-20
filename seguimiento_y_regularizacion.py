# Commented out IPython magic to ensure Python compatibility.
# %pip install comet_ml

import comet_ml
comet_ml.init(project_name="Prueba comet")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint

experiment = comet_ml.Experiment(
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    log_code=True,
)

dataset=mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # x_trainv = x_trainv/255
x_test /= 255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

parameters = {
    "batch_size": 128,
    "epochs": 20,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
}

experiment.log_parameters(parameters)

model = Sequential()
model.add(Input(shape=(28,28))) #Flaten no tiene la opcion input_shape por lo tanto se tiene que agregar esta capa
model.add(Flatten()) #Otra forma de aplanar las imagenes
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))) #Regularizacion L1L2
model.add(Dropout(0.2)) #Fraccion de enlaces a eliminar
model.add(Dense(200)) #Capa lineal , transformacion lineal sin funcion de activacion
model.add(Activation('tanh')) #Se puede agregar despues la funcion de activacion
model.add(Dense(400, activation='selu', kernel_regularizer=regularizers.L1(0.01) )) #Regularizacion L1
model.add(Dense(200, activation='elu', kernel_regularizer=regularizers.L2(l2=1e-4)) ) #Regularizacion L2
model.add(Dense(50,activation='exponential'))
model.add(Dense(10, activation='softmax'))

model.summary()

# specify the path where you want to save the model
filepath = "best_model.hdf5"

# initialize the ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.compile(loss=parameters['loss'],optimizer=parameters['optimizer'],metrics=['accuracy'])
model.fit(x_train, y_trainc,
                    batch_size=parameters['batch_size'],
                    epochs=parameters["epochs"],
                    verbose=1,
                    validation_data=(x_test, y_testc),
                    callbacks=[checkpoint])

experiment.log_model("MNIST1", "best_model.hdf5")

experiment.end()