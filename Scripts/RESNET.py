# import cv2
# import numpy as np
# import os
# from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
# import keras
# from keras.models import Sequential, Model,load_model
# from keras.optimizers import SGD
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from google.colab.patches import cv2_imshow
# from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
# from keras.preprocessing import image
# from keras.initializers import glorot_uniform


#####################MYCODE############################################################################
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
import csv
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras import layers
#from tensorflow.keras.utils import plot_model

tf.compat.v1.enable_eager_execution()

file_predicted_labels = open("prediction-result1.csv", "w")
file_test_data = open("test-data1.csv","w")
file_test_labels = open("test-labels1.csv","w")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import  Model
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, BatchNormalization, Conv1D,MaxPooling1D,Conv2D
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow import keras
import numpy as np
#Load ECG data
#The dataset you will use is based on one from timeseriesclassification.com.
import pandas as pd

import tensorflow as tf
from tensorflow.compat.v1 import Session as sess
#tf.compat.v1.disable_eager_execution()

# Download the dataset
dataframe = pd.read_csv('sEMG_prediction_dataset(channel_and_3Dkeypoints).csv')
raw_data = pd.DataFrame(dataframe)#dataframe.values
raw=raw_data.values



# The last element contains the labels

labels=raw[:,9:27].astype(float)
print(labels)
#y_test = dfTest_dataValues[:, len(dfTest_data.columns) - 1]\n

#labels=labels[:,-1].values

y=labels
print(y)
# The other data points are the electrocadriogram data
data = raw[:, 1:9].astype(float)




train_data2 = data.reshape(data.shape[0], data.shape[1],1).astype( 'float32')
print('train_data_shape_original:', train_data2.shape)
print(y.shape)
#y=labels.reshape(labels.shape[0],labels.shape[1],1).astype('float32')
#y = tf.squeeze(y, axis=-1)
#print('new y',y)
#y= y.reshape(data.shape[0], 3, 6).astype( 'float32')
print('y_data_shape:', y.shape)
#print(y)

train_data, val_data, train_labels, val_labels = train_test_split( train_data2,y, test_size=0.3 )
print('train_data_shape2:', train_data.shape)

train_data, test_data, train_labels, test_labels = train_test_split(train_data,train_labels, test_size=0.2 )
print('train_data_shape2:', train_data.shape)

#
# import random
#
# # Split our img paths into a training and a validation set
# val_samples = 1000
# random.Random(1337).shuffle(input_img_paths)
# random.Random(1337).shuffle(target_img_paths)
# train_input_img_paths = input_img_paths[:-val_samples]
# train_target_img_paths = target_img_paths[:-val_samples]
# val_input_img_paths = input_img_paths[-val_samples:]
# val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split






#print('train_data, test_data, train_labels, test_labels')

#print(train_data, test_data, train_labels, test_labels)
#print('shape of train data',train_data2.shape)

img_size = ( 1,8,1)

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
    x = layers.Conv2D(num, 3, activation="softmax", padding="same")(x)
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

#input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
# x = tf.keras.layers.Embedding(
#     output_dim=512, input_dim=10000, input_length=100)(input)
# x = tf.keras.layers.LSTM(32)(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
# model = tf.keras.Model(inputs=[input], outputs=[output])
# dot_img_file = '/tmp/model_1.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)




loss = tf.keras.losses.Huber(delta=1.05)
# Build model
from tensorflow.keras.models import Sequential
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

x_train = train_data.reshape(-1,1, 8, 1)   #Reshape for CNN -  should work!!
val_data = val_data.reshape(-1,1, 8, 1)
#x_labels = train_labels.reshape(-1,1, 18)


import numpy as np

x_train = np.asarray(x_train)
train_labels = np.asarray(train_labels)
val_data = np.asarray(val_data)
val_labels = np.asarray(val_labels)

#history_model_1=model.fit(x_train, train_labels, epochs=1,validation_data=(val_data,val_labels))#, validation_data=(x_test, y_test))
history_model_1=model.fit(
    x_train,
    train_labels,
    epochs=1,
    validation_data=(val_data, val_labels)
)
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
        diffA = real - predict
        diff = np.absolute(real-predict)
        diffMean = np.mean(diff)
        diffvar = np.var(diff)
        diffstd = np.std(diff)
        diffmin = np.min(diff)
        diffmax = np.max(diff)

        #percentage =  1-np.sqrt(np.mean(np.square((real - predict) )))
        rmse =  np.sqrt(np.mean(np.square(real - predict )))
        percentage2 = np.mean(np.sqrt(np.square(real - predict)))
        percentage3 = np.mean(np.square((real - predict)))

        # import pandas as pd
        # pred_dfError = pd.DataFrame(diffA)
        # pred_dfError.to_csv('trap.csv')

        return rmse, percentage2, percentage3, diffMean, diffvar, diffstd, diffmin, diffmax


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


# Train the model, doing validation at the end of each epoch.

#model.fit(train_data, train_labels,epochs=100)
#model.fit(train_data,train_labels, epochs=15)

#model.fit(training,epochs=15 )


# base_model = ResNet50(input_shape=(224, 224, 3))
#
# headModel = base_model.output
#
# headModel = Flatten()(headModel)
# headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
# headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
# headModel = Dense( 1,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)
# model = Model(inputs=base_model.input, outputs=headModel)
#
# model.summary()
# base_model.load_weights("/content/gdrive/My Drive/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
#
# for layer in base_model.layers:
#     layer.trainable = False
# for layer in model.layers:
#     print(layer, layer.trainable)
#
# es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
#
# H = model.fit_generator(train_generator,validation_data=test_generator,epochs=100,verbose=1,callbacks=[mc,es])
# model.load_weights("/content/gdrive/My Drive/best_model.h5")
# model.evaluate_generator(test_generator)
# model_json = model.to_json()
# with open("/content/gdrive/My Drive/model.json","w") as json_file:
#   json_file.write(model_json)
#
# from keras.models import model_from_json
#
# def predict_(image_path):
#     # Load the Model from Json File
#     json_file = open('/content/gdrive/My Drive/model.json', 'r')
#     model_json_c = json_file.read()
#     json_file.close()
#     model_c = model_from_json(model_json_c)
#     # Load the weights
#     model_c.load_weights("/content/gdrive/My Drive/best_model.h5")
#     # Compile the model
#     opt = SGD(lr=1e-4, momentum=0.9)
#     model_c.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#     # load the image you want to classify
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224))
#     cv2_imshow(image)
#     # predict the image
#     preds = model_c.predict(np.expand_dims(image, axis=0))[0]
#     if preds == 0:
#         print("Predicted Label:Cat")
#     else:
#         print("Predicted Label: Dog")
#
#
# predict_("/content/gdrive/My Drive/datasets/test/Dog/4.jpg")
#
# predict_("/content/gdrive/My Drive/datasets/test/Cat/10.jpg")
#
# # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
# #
# # def identity_block(X, f, filters, stage, block):
# #
# #     conv_name_base = 'res' + str(stage) + block + '_branch'
# #     bn_name_base = 'bn' + str(stage) + block + '_branch'
# #     F1, F2, F3 = filters
# #
# #     X_shortcut = X
# #
# #     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
# #     X = Activation('relu')(X)
# #
# #     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
# #     X = Activation('relu')(X)
# #
# #     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
# #
# #     X = Add()([X, X_shortcut]  )# SKIP Connection
# #     X = Activation('relu')(X)
# #
# #     return X
# #
# #
# # def convolutional_block(X, f, filters, stage, block, s=2):
# #     conv_name_base = 'res' + str(stage) + block + '_branch'
# #     bn_name_base = 'bn' + str(stage) + block + '_branch'
# #
# #     F1, F2, F3 = filters
# #
# #     X_shortcut = X
# #
# #     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
# #                kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
# #     X = Activation('relu')(X)
# #
# #     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
# #                kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
# #     X = Activation('relu')(X)
# #
# #     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
# #                kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
# #
# #     X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
# #                         kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
# #     X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
# #
# #     X = Add()([X, X_shortcut])
# #     X = Activation('relu')(X)
# #
# #     return X
# #
# #
# # def ResNet50(input_shape=(224, 224, 3)):
# #     X_input = Input(input_shape)
# #
# #     X = ZeroPadding2D((3, 3))(X_input)
# #
# #     X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
# #     X = BatchNormalization(axis=3, name='bn_conv1')(X)
# #     X = Activation('relu')(X)
# #     X = MaxPooling2D((3, 3), strides=(2, 2))(X)
# #
# #     X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
# #     X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
# #     X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
# #
# #     X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
# #     X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
# #     X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
# #     X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
# #
# #     X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
# #     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
# #     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
# #     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
# #     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
# #     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
# #
# #     X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
# #     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
# #
# #     X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
# #
# #     model = Model(inputs=X_input, outputs=X, name='ResNet50')
# #
# #     return model
