#Architecture NN:
# 1024 X dropout(0.5) X 2048 X 41 (result(val/test): (0,98/ 0,8147)

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D,Conv1D, MaxPool2D, regularizers
from keras import backend as K
from clr_callback import CyclicLR
import termcolor

class LRTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        # logs.update({'val/loss': K.eval(self.model.fit)})
        super().on_epoch_end(epoch, logs)

import tensorflow as tf
import termcolor
import glob

# Generate dummy data
import numpy as np

output_class = 41

class VggDNN(object):
    def __init__(self, path = None,input_shape=(128,), lr = 0.0004, optimizer = 'SGD'):
        if path != None:
            self.model = self.load_model(path)
        else:
            self.model = self.create_model(input_shape=input_shape, lr=lr, optimizator=optimizer)

    def create_model(self, input_shape=(128,), output = 41, lr = 0.0004, optimizator = 'SGD'):

        try:
            model = Sequential()
            model.add(Dense(2048, activation='relu',kernel_initializer='glorot_uniform', input_shape=input_shape))
            model.add(keras.layers.BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu',kernel_initializer='glorot_uniform'))
            model.add(Dropout(0.5))
            model.add(Dense(41, activation='softmax',kernel_initializer='glorot_uniform'))


            self.start_lr = lr
            self.lr = float(self.start_lr)
            self.count_lr = int(0)

            if optimizator == 'SGD':
                print(termcolor.colored("SGD","yellow"))
                optimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
            elif optimizator == 'adam':
                print(termcolor.colored("adam","yellow"))
                optimizer = keras.optimizers.adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon= None, decay=0.0, amsgrad=False);
            elif optimizator == 'adamax':
                print(termcolor.colored("adamax","yellow"))
                optimizer = keras.optimizers.adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon= None, decay=0.0);
            else:
                print(termcolor.colored("SGD","yellow"))
                optimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)

            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        except:
            raise Exception("create_model() is failed");

        return model

    def train(self, train, valid, checkPath, epochs = 200, factor = 0.8, batch_size = 32, tensorboardPath = None, lim_lr = 0.0009, scheduler_mode = None, iteration = None):
        try:
            if tensorboardPath == None:

                t,h = os.path.split(checkPath)
                tensorboardPath = os.path.normpath(t + "//tensorboard//"+str(h))

            patience_stop = 100
            patience_reduce = 3

            self.factor = factor

            earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_stop)
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=checkPath, verbose=1, save_best_only=True)
            tboard = LRTensorBoard(tensorboardPath)

            if scheduler_mode == None:
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=float(factor), patience=patience_reduce, min_delta=0.2, min_lr=lim_lr )
            else:
                str_info = ' scheduler_mode: ' + str(scheduler_mode) + ','
                str_info += ' base_lr: ' + str(self.start_lr) + ','
                str_info += ' max_lr ' + str(self.start_lr * float(factor)) + ','
                str_info += ' step_size: '+ str(int(2*iteration))
                print(termcolor.colored(str_info,"yellow"))

                if iteration == None:
                    raise Exception("Number of iteration is unknown")

                reduce_lr = CyclicLR(base_lr=self.start_lr, max_lr=self.start_lr * float(factor), mode=scheduler_mode, step_size=2 * iteration)

            callback_list = [earlystop, checkpointer, tboard, reduce_lr]

            self.model.fit_generator(generator=train, validation_data=valid, epochs=epochs, use_multiprocessing=True, callbacks=callback_list,)

            del self.model
        except:
            raise Exception("train is failed")

    def predict(self, x,batch_size = 32):
        try:
            return self.model.predict(x,batch_size=batch_size)
        except:
            raise Exception("train is failed")

    def predict_on_batch(self,x):
        return self.model.predict_on_batch(x)

    def load_model(self, model):
        try:
            # model = os.path.normpath(model)
            # status_string = "Loade model: " + model
            # print(termcolor.colored(status_string,"yellow"))

            return keras.models.load_model(filepath=model)
        except:
            raise  Exception("Load model is failed");


    def _schedule(self, epoch):

        lr = self.lr
        count_lr = self.count_lr


        if count_lr >= 0 and count_lr < 5:
            lr += self.factor
        elif count_lr >= 5 and count_lr < 10:
            lr -= self.factor
        elif count_lr > 10:
            self.count_lr = int(0)
            self.lr = self.start_lr
        else:
            lr = self.lr

        self.count_lr += int(1)

        self.lr = lr

        print("Epoch: ",str(epoch)," | Lr: ",str(lr))

        return float(lr)