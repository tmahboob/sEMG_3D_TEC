
import csv


#from sgdr import SGDRScheduler
import time
file_predicted_labels = open("prediction-result1.csv", "w")
file_test_data = open("test-data1.csv","w")
file_test_labels = open("test-labels1.csv","w")

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import  Model
import keras
from tensorflow import keras
import numpy as np
#Load ECG data
#The dataset you will use is based on one from timeseriesclassification.com.
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import Session as sess
tf.compat.v1.disable_eager_execution()

#dataframe = pd.read_csv('train-11242021.csv')
#dataframe = pd.read_csv('filterd_50_1.csv')
#dataframe = pd.read_csv('idea-feb15..csv')
dataframe = pd.read_csv('DATASET-16FEB2022.csv')
raw_data = pd.DataFrame(dataframe)#dataframe.values
raw=raw_data.values


# The last 18 element contains the labels
labels=raw[:,9:27].astype(float)
print(labels)
#y_test = dfTest_dataValues[:, len(dfTest_data.columns) - 1]\n
#labels=labels[:,-1].values

y=labels
print(y)
# The first 8 columns are the electrocadriogram 8 channel data
data = raw[:, 1:9].astype(float)

#reshaping the data to feed the Network
train_data2 = data.reshape(data.shape[0], data.shape[1],1).astype( 'float32')
print('train_data_shape_original:', train_data2.shape)
print(y.shape)
#y=labels.reshape(labels.shape[0],labels.shape[1],1).astype('float32')
#y = tf.squeeze(y, axis=-1)
#print('new y',y)
#y= y.reshape(data.shape[0], 3, 6).astype( 'float32')
print('y_data_shape:', y.shape)
#print(y)

#Splitting the data into training, validation, and test sets

train_data, val_data, train_labels, val_labels = train_test_split(train_data2,y, test_size=0.3)
print('train_data_shape2:', train_data.shape)

train_data, test_data, train_labels, test_labels = train_test_split(train_data,train_labels, test_size=0.2)
print('train_data_shape2:', train_data.shape)

###input_shape, output classes
img_size = (1,8,1)
num=18



def getmodel(img_size,num):
    input_shape = img_size
    inputs = keras.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###
    # Entry block
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    x = layers.Conv2D(num, 3, activation="relu", padding="same")(x)
    #x=layers.add(Flatten())
    #x=layers.add(Dense(256, activation='relu'))
   # x = layers.Conv2D(num_classes,3, activation="softmax", padding="same")
   #y = tf.keras.layers.Add()([x1, x2])
   # c = layers.Activation("linear")(x)
    t = layers.Flatten()(x)
    s=layers.Dense(256,activation='relu')(t)
    outputs = layers.Dense(num)(s)# .add(Dense(18))  # model.add(Dense(256, activation = 'relu'))

    # Define the model
    model = Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

model= getmodel(img_size,num)
model.summary()
#plot_model(getmodel, to_file='model.png', show_shapes=True, show_layer_names=True)


loss = tf.keras.losses.Huber(delta=1.05)
# Build model
#from tf.keras.optimzers import Adam
#adam=keras.optimizers.Adam(lr=0.01)
model.compile(optimizer="adam", loss=loss)

# callbacks = [
#     keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
# ]

#from keras.callbacks import Callback
import os

# Get the current working directory


input_shape = img_size
inputs = keras.Input(shape=input_shape)

#training=(32,img_size, train_data, train_labels)
x_train = train_data.reshape(-1,1,8, 1)   #Reshape for CNN -  should work!!
val_data = val_data.reshape(-1,1,8, 1)
#x_labels = train_labels.reshape(-1,32,1, 18)

history_model_1=model.fit(x_train, train_labels, epochs=1
                          ,validation_data=(val_data,val_labels))#, validation_data=(x_test, y_test))

import pandas as pd

pd.DataFrame.from_dict(history_model_1.history).to_csv('history.csv',index=False)


x_test = test_data.reshape(-1,1, 8, 1)
print("*******************EVALUATE MODEL***************************")
loss = model.evaluate(x_test, test_labels)
print("Loss:", loss)
# predictions = model.predict(x_test)
# print("predictions shape:", predictions.shape)


writer = csv.writer(file_test_data, lineterminator='\n')
attributes_test_data = (x_test)
writer.writerow(attributes_test_data)
print("*******************PREDICT MODEL***************************")
pred = model.predict(x_test)
#print('pred', pred)

writer = csv.writer(file_predicted_labels, lineterminator='\n')
attributes_predicted_labels = (pred)
writer.writerow(attributes_predicted_labels)

writer = csv.writer(file_test_labels, lineterminator='\n')
attributes_test_labels = (test_labels)
writer.writerow(attributes_test_labels)

#pred_y = pred.argmax(axis=-1)
#print('pred_y',pred_y)
#y_test=np.argmax(test_labels,axis=-1)
#print('y_test',y_test)
# cm = confusion_matrix(test_labels, pred)
# print(cm)


#y_pred=model.predict(X_test)
#y_pred=np.argmax(y_pred, axis=1)
#y_test=np.argmax(y_test, axis=1)
#cm = confusion_matrix(y_test, y_pred)

def calculate_accuracy(real, predict):
        real = np.array(real) +1
        predict = np.array(predict) +1
        predict=(predict)
        #diffA = real - predict
        #diff = np.absolute(real-predict)
        #diffMean = np.mean(diff)
        #diffvar = np.var(diff)
        #diffstd = np.std(diff)
        #diffmin = np.min(diff)
        #diffmax = np.max(diff)

        #percentage =  1-np.sqrt(np.mean(np.square((real - predict) )))
        rmse =  np.sqrt(np.mean(np.square(real - predict )))



        #percentage2 = np.mean(np.sqrt(np.square(real - predict)))
        #percentage3 = np.mean(np.square(np.square(real - predict)))

        # import pandas as pd
        # pred_dfError = pd.DataFrame(diffA)
        # pred_dfError.to_csv('trap.csv')

        return rmse, # diffMean, diffvar, diffstd, diffmin, diffmax


print('test labels',test_labels)
import pandas as pd
pred_df= pd.DataFrame(pred)
pred_df.to_csv('pref1.csv')
predicted_value = pred_df.iloc[:, :].values
#predicted_value= pred_df.iloc[:, 0:1].dropna()
print('length of predicted values csv file:',len(predicted_value))

import pandas as pd
labels_df= pd.DataFrame(test_labels)
labels_df.to_csv('labels.csv')
real_value = labels_df.iloc[:, :].values
#predicted_value= pred_df.iloc[:, 0:1].dropna()
print('length of labels csv file:',len(real_value))

#Accuracy_result=calculate_accuracy(real_value[:len(predicted_value),:], (predicted_value[:len(predicted_value),:]))
Accuracy_result=calculate_accuracy(real_value[1:,:], (predicted_value[1:,:]))
print(real_value[1])
print(predicted_value[1])
print('RMSE1''RMSE2','MSE', 'Mean-MAE,','Variance-AE,','STDEV-AE,','Errormin','Errormax')
print(Accuracy_result)
