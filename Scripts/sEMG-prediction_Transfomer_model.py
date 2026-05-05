######
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

import keras

from tensorflow.keras.layers import Reshape
# Enable eager execution (although it should be on by default in TensorFlow 2.x)
#tf.config.run_functions_eagerly(True)
file_predicted_labels = open("prediction-result1.csv", "w")
file_test_data = open("test-data1.csv","w")
file_test_labels = open("test-labels1.csv","w")
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import  Model

from tensorflow import keras

from tensorflow.compat.v1 import Session as sess
from keras.layers import Dense, SimpleRNN, Reshape
#tf.compat.v1.disable_eager_execution()

dataframe = pd.read_csv('sEMG_prediction_dataset(channel_and_3Dkeypoints).csv')
#dataframe = pd.read_csv('combined_data.csv')



raw_data = pd.DataFrame(dataframe)
raw = raw_data.values

raw = raw.astype(np.float32)

# Replace NaNs / infs
raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize (VERY important)
mean = raw.mean(axis=0)
std = raw.std(axis=0) + 1e-8
raw = (raw - mean) / std

num_channels = 27
data_records = 41
sliding_window = 10

################################Feb 23, 2022####################
data=[]
y=[]
#data_records=16
for i in range(data_records, len(raw),sliding_window):
    data.append(raw[i-data_records:i, 0:27])
    y.append(raw[i,:])#,0:1]


data, y = np.array(data), np.array(y)
x_train, x_test, y_train, y_test = train_test_split(data,y, test_size=0.2)
print('train_data_shape2:', x_train.shape)

img_size = (data_records, num_channels)
n_classes = (1)
print("LENGTH OF XTRAIN, Y TRAIN, X TEST, Y TEST", len(x_train),len(y_train), len(x_test), len(y_test))

###################################


import numpy as np
# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from tensorflow import keras
from tensorflow.keras import layers

#vocab_size = 1000 # Only consider the top 20k words
#maxlen = 8
#embed_dim = 8
img_size = (data_records, num_channels)
#input_shape = x_train.shape[1:]
input_shape = img_size

maximum_position_encoding =22
d_model =64
#dff = 4


embedding_dim = 64
import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import  Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


    # create a simple embedding layer with sinusoidal positional encoding


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    print (angle_rads)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)




class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=pow(10,-6))(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=pow(10,-6))(res)
    x = layers.Dense(ff_dim)(x)
    x = layers.Dropout(dropout)(x)
    print('WHATT', inputs.shape[-1])
    x = layers.Dense( inputs.shape[-1])(x)
    return x + res


#def transformer_encoder1(inputs, head_size, num_heads, ff_dim, dropout=0):
   # x = layers.LayerNormalization(epsilon=1e-6)(inputs)
  #  x = layers.MultiHeadAttention(
  #      key_dim=head_size, num_heads=num_heads, dropout=dropout
 #   )(x, x)
#    x = layers.Dropout(dropout)(x)
#    res = x + inputs
  #  x = layers.LayerNormalization(epsilon=1e-6)(res)
  #  x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
 #   x = layers.Dropout(dropout)(x)
#    return x + res




import numpy as np
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape) #input_shape = (data_records, num_channels)
    x = inputs
    x = tf.keras.layers.Conv1D(64, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv1D(128, 1, strides=2, padding="same")(x)


    # Assuming `x` is the output of a previous layer


    #x = tf.keras.layers.Conv1D(256, 1, strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    print('x here is after dense', x)
    #x = Reshape((22, 64))(x)
    x = Reshape((22, 64))(x)
    #x = tf.reshape(x, [1,256, 64])

    print('modellllllllllllll', x.shape)
    print('x after reshape is', x)



    #print('tf.shape(x)[1]',tf.shape(x)[1])
    pos = positional_encoding(maximum_position_encoding, d_model)
    print('position encoding is',pos)
    x = tf.keras.layers.Add()([x, pos])


    #x = tf.keras.layers.Add()([x, pos[:, :tf.shape(x)[1], :]])
    for i in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    print('x here before GAP is', x)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) #If data_format='channels_first': 4D tensor with shape (batch_size, channels, rows, cols)./
    print('x after global average pooling', x)                                                                   #//If data_format='channels_last': 4D tensor with shape (batch_size, rows, cols, channels)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

   # x = tf.reshape(x, [-1, 8, 1])
   # x = layers.LSTM(units=1, return_sequences=False, input_shape=(1, 8, 1)) (x)
    outputs = layers.Dense(n_classes,activation='sigmoid')(x)

    return keras.Model(inputs, outputs)
#


#print('transformer input', input_shape)
# input_shape = img_size
# inputs = keras.Input(shape=input_shape)

model = build_model(
    input_shape,
    head_size = 22,
    num_heads = 4,
    ff_dim = 64,
    num_transformer_blocks=2,
    mlp_units = [128],
    mlp_dropout = 0.4,
    dropout = 0.25
)

loss = tf.keras.losses.Huber(delta=0.7)
# Build model
model.compile(optimizer = keras.optimizers.Adam(learning_rate=pow(10,-4)), loss=loss)

# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#     metrics=["sparse_categorical_accuracy"],
# )
model.summary()

# = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history=model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=1)
 #   callbacks=callbacks,
#)



#y_test= y_test.reshape(-1,8,1)
x = x_test


print('Shape of x_test and y_test', x, y_test.shape)
eva=model.evaluate(x, y_test, batch_size=1)#, verbose=1)
print("Evaluation result:", eva)


import pandas as pd

pd.DataFrame.from_dict(history.history).to_csv('history.csv', index=False)



#test_labels = test_labels.reshape(-1,1,16,18)
print("*******************EVALUATE MODEL***************************")
#loss = model.evaluate(x_test, test_labels)
print("Loss:", loss)
# predictions = model.predict(x_test)
# print("predictions shape:", predictions.shape)


writer = csv.writer(file_test_data, lineterminator='\n')
attributes_test_data = (x_test)
writer.writerow(attributes_test_data)
print("*******************PREDICT MODEL***************************")
pred = model.predict(x_test,batch_size=1)


#print('pred', pred)

writer = csv.writer(file_predicted_labels, lineterminator='\n')
attributes_predicted_labels = (pred)
writer.writerow(attributes_predicted_labels)

writer = csv.writer(file_test_labels, lineterminator='\n')
attributes_test_labels = (y_test)
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
count=0

####in correct accuracy function#####
def calculate_accuracy1(real, predict):
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

        #percentage =  1-np.sqrt(np.mean(np.square((real - predict) ))
        threshold=30
        rmse =  np.sqrt(np.mean(np.square(real - predict)))
        print('RMSE :', rmse)
        count=0
        for i in range(1,len(real)):
            if  np.mean(np.sqrt((np.square(real[i,1:2]-predict[i,1:2])))) < threshold:
                count= count+1
                if np.mean(np.sqrt((np.square(real[i,2:3]-predict[i,2:3])))) < threshold:
                    count = count + 1
                    if np.mean(np.sqrt((np.square(real[i, 3:4] - predict[i, 3:4])))) < threshold:
                        count = count + 1
                        if np.mean(np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5])))) < threshold:
                            count = count + 1
                            if np.mean(np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6])))) < threshold:
                                count = count + 1
                                if np.mean(np.sqrt((np.square(real[i, 6:7] - predict[i, 6:7])))) < threshold:
                                    count = count + 1
                                    if np.mean(np.sqrt((np.square(real[i, 7:8] - predict[i, 7:8])))) < threshold:
                                        count = count + 1
                                        if np.mean(np.sqrt((np.square(real[i, 8:9] - predict[i, 8:9])))) < threshold:
                                            count = count + 1
                                            if np.mean(np.sqrt((np.square(real[i, 9:10] - predict[i, 9:10])))) < threshold:
                                                count = count + 1
                                                if np.mean(np.sqrt((np.square(real[i, 10:11] - predict[i, 10:11])))) < threshold:
                                                    count = count + 1
                                                    if np.mean(np.sqrt((np.square(real[i, 11:12] - predict[i, 11:12])))) < threshold:
                                                        count = count + 1
                                                        if np.mean(np.sqrt((np.square(real[i, 12:13] - predict[i, 12:13])))) < threshold:
                                                            count = count + 1
                                                            if np.mean(np.sqrt((np.square(real[i, 13:14] - predict[i, 13:14])))) < threshold:
                                                                count = count + 1
                                                                if np.mean(np.sqrt((np.square(real[i, 14:15] - predict[i, 14:15])))) < threshold:
                                                                    count = count + 1
                                                                    if np.mean(np.sqrt((np.square(real[i,15:16] - predict[i, 15:16])))) < threshold:
                                                                        count = count + 1
                                                                        if np.mean(np.sqrt((np.square(real[i, 16:17] - predict[i, 16:17])))) < threshold:
                                                                            count = count + 1
                                                                            if np.mean(np.sqrt((np.square(real[i, 17:18] - predict[i, 17:18])))) < threshold:
                                                                                count = count + 1
                                                                                if np.mean(np.sqrt((np.square(real[i, 18:19] - predict[i, 18:19])))) < threshold:
                                                                                    count = count + 1


              #print("rmse for 1st is:", np.sqrt((np.square(real[2]-predict[2]))/6))
            print("count", count)


        #percentage2 = np.mean(np.sqrt(np.square(real - predict)))
        #percentage3 = np.mean(np.square(np.square(real - predict)))
        # import pandas as pd
        # pred_dfError = pd.DataFrame(diffA)
        # pred_dfError.to_csv('trap.csv')

        return rmse, count/(np.multiply(18,len(real)))#)len(real) # diffMean, diffvar, diffstd, diffmin, diffmax

def calculate_accuracy(real, predict):
    threshold = 30
    rmse = np.sqrt(np.mean(np.square(real - predict)))
    print('RMSE :', rmse)
    count = 0
    for i in range(1, len(real)):
        if np.sqrt((np.square(real[i, 0:1] - predict[i, 0:1])) + (np.square(real[i, 6:7] - predict[i, 6:7])) + (
           np.square(real[i, 12:13] - predict[i, 12:13]))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print((np.sqrt((np.square(real[i, 0:1] - predict[i, 0:1])) + (np.square(real[i, 6:7] - predict[i, 6:7])) + (
                #np.square(real[i, 12:13] - predict[i, 12:13])))))
            #print('thumb count for', 'sample numnber: ', i, count)
        else:
            print("thumb count not updated for iteration: ", i, "and count remains", count)
        if (np.sqrt((np.square(real[i, 1:2] - predict[i, 1:2])) + (np.square(real[i, 7:8] - predict[i, 7:8])) + (
            np.square(real[i, 13:14] - predict[i, 13:14])))) < threshold:
            count = count + 1
            #print(np.sqrt((np.square(real[i, 1:2] - predict[i, 1:2])) + (np.square(real[i, 7:8] - predict[i, 7:8])) + (
                #np.square(real[i, 13:14] - predict[i, 13:14]))))
            #print('threshold:', threshold)
            #print('index count:', 'sample numnber: ', i, count)
        else:
            print("index count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
            np.square(real[i, 14:15] - predict[i, 14:15])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
                #np.square(real[i, 14:15] - predict[i, 14:15]))))
            #print('middle count:', 'sample numnber: ', i, count)
        else:
            print("middle count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 3:4] - predict[i, 3:4])) + (np.square(real[i, 9:10] - predict[i, 9:10])) + (
        np.square(real[i, 15:16] - predict[i, 15:16])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
                #np.square(real[i, 14:15] - predict[i, 14:15]))))
            #print('ring count:''sample numnber: ', i, count)
        else:
            print("ring count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5])) + (np.square(real[i, 10:11] - predict[i, 10:11])) + (
        np.square(real[i, 16:17] - predict[i, 16:17])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5])) + (np.square(real[i, 10:11] - predict[i, 10:11])) + (
                #np.square(real[i, 16:17] - predict[i, 16:17]))))
            #print('pinky count:', 'sample numnber: ', i, count)
        else:
            print("pinky count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6])) + (np.square(real[i, 11:12] - predict[i, 11:12])) + (
        np.square(real[i, 17:18] - predict[i, 17:18])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6])) + (np.square(real[i, 11:12] - predict[i, 11:12])) + (
                #np.square(real[i, 17:18] - predict[i, 17:18]))))
            #print('palm count:', 'sample numnber: ', i, count)
        else:
            print("palm count not updated for iteration: ", i, "and count remains", count)

        # print("rmse for 1st is:", np.sqrt((np.square(real[2]-predict[2]))/6))
    print("Count Final: ", count)

    # percentage2 = np.mean(np.sqrt(np.square(real - predict)))
    # percentage3 = np.mean(np.square(np.square(real - predict)))
    # import pandas as pd
    # pred_dfError = pd.DataFrame(diffA)
    # pred_dfError.to_csv('trap.csv')
    print ('length of real',len(real))
    return rmse, count / (np.multiply(6, len(real)))  # )len(real) # diffMean, diffvar, diffstd, diffmin, diffmax


def calculate_accuracy_1D(real, predict):
    threshold = 30
    rmse = np.sqrt(np.mean(np.square(real - predict)))
    print('RMSE :', rmse)
    count = 0
    for i in range(1, len(real)):
        if (np.sqrt(np.square(real[i, 0:1] - predict[i, 0:1])))< threshold:
                   # + (np.square(real[i, 6:7] - predict[i, 6:7])) + (
       # np.square(real[i, 12:13] - predict[i, 12:13])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print((np.sqrt((np.square(real[i, 0:1] - predict[i, 0:1])) + (np.square(real[i, 6:7] - predict[i, 6:7])) + (
                #np.square(real[i, 12:13] - predict[i, 12:13])))))
            #print('thumb count for', 'sample numnber: ', i, count)
        else:
            print("thumb count not updated for iteration: ", i, "and count remains", count)
        if (np.sqrt(np.square(real[i, 1:2] - predict[i, 1:2])))< threshold:
                    #+ (np.square(real[i, 7:8] - predict[i, 7:8])) + (
           # np.square(real[i, 13:14] - predict[i, 13:14])))) < threshold:
            count = count + 1
            #print(np.sqrt((np.square(real[i, 1:2] - predict[i, 1:2])) + (np.square(real[i, 7:8] - predict[i, 7:8])) + (
                #np.square(real[i, 13:14] - predict[i, 13:14]))))
            #print('threshold:', threshold)
            #print('index count:', 'sample numnber: ', i, count)
        else:
            print("index count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) ))<threshold:
                    #+ (np.square(real[i, 8:9] - predict[i, 8:9])) + (
           # np.square(real[i, 14:15] - predict[i, 14:15])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
                #np.square(real[i, 14:15] - predict[i, 14:15]))))
            #print('middle count:', 'sample numnber: ', i, count)
        else:
            print("middle count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 3:4] - predict[i, 3:4]))))<threshold:
                    #+ (np.square(real[i, 9:10] - predict[i, 9:10])) + (
        #np.square(real[i, 15:16] - predict[i, 15:16])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
                #np.square(real[i, 14:15] - predict[i, 14:15]))))
            #print('ring count:''sample numnber: ', i, count)
        else:
            print("ring count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5]))))<threshold:
                    #+ (np.square(real[i, 10:11] - predict[i, 10:11])) + (
        #np.square(real[i, 16:17] - predict[i, 16:17])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5])) + (np.square(real[i, 10:11] - predict[i, 10:11])) + (
                #np.square(real[i, 16:17] - predict[i, 16:17]))))
            #print('pinky count:', 'sample numnber: ', i, count)
        else:
            print("pinky count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6]))))<threshold:
                    #+ (np.square(real[i, 11:12] - predict[i, 11:12])) + (
        #np.square(real[i, 17:18] - predict[i, 17:18])))) < threshold:
            count = count + 1
            #print('threshold:', threshold)
            #print(np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6])) + (np.square(real[i, 11:12] - predict[i, 11:12])) + (
                #np.square(real[i, 17:18] - predict[i, 17:18]))))
            #print('palm count:', 'sample numnber: ', i, count)
        else:
            print("palm count not updated for iteration: ", i, "and count remains", count)

        # print("rmse for 1st is:", np.sqrt((np.square(real[2]-predict[2]))/6))
    print("Count Final: ", count)

    # percentage2 = np.mean(np.sqrt(np.square(real - predict)))
    # percentage3 = np.mean(np.square(np.square(real - predict)))
    # import pandas as pd
    # pred_dfError = pd.DataFrame(diffA)
    # pred_dfError.to_csv('trap.csv')

    return rmse, count / (np.multiply(6, len(real)))  # )len(real) # diffMean, diffvar, diffstd, diffmin, diffmax




print('test labels',y_test)
import pandas as pd
pred_df= pd.DataFrame(pred)
pred_df.to_csv('pref1.csv')
predicted_value = pred_df.iloc[:, :].values
#predicted_value= pred_df.iloc[:, 0:1].dropna()
print('length of predicted values csv file:',len(predicted_value))

import pandas as pd
labels_df= pd.DataFrame(y_test)
labels_df.to_csv('labels.csv')
real_value = labels_df.iloc[:, :].values
#predicted_value= pred_df.iloc[:, 0:1].dropna()
print('length of labels csv file:',len(real_value))

#Accuracy_result=calculate_accuracy(real_value[:len(predicted_value),:], (predicted_value[:len(predicted_value),:]))
Accuracy_result=calculate_accuracy(real_value[1:len(real_value),:], (predicted_value[1:len(real_value),:]))
print(real_value[5])
print(predicted_value[5])
print('RMSE1')#'RMSE2','MSE', 'Mean-MAE,','Variance-AE,','STDEV-AE,','Errormin','Errormax')
print(Accuracy_result)
#pck=[]
#if Accuracy_result < 5:
#    pck = rmse / len(real_value[1:, 0:1])
#print("PCK is: ", pck)

import matplotlib.pyplot as plt

fig = plt.figure(1)

#ax = plt.axes()
ax2 =plt.axes()

y1=real_value[1:,0:1]
y2=predicted_value[1:,0:1]
x= range(1, len(real_value))





plt.plot(x[1:100], y1[1:100], 'bo')
plt.plot(x[1:100], y2[1:100], 'go')

ax2.legend()
ax2.set_xlabel('x-axis')
ax2.set_ylabel('y-axis')

plt.show()
####################### 3D image ######################
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
fig = plt.figure(2)
ax = plt.axes(projection='3d')



xline=real_value[1:2,0:6]
yline=real_value[1:2,6:12]
zline=real_value[1:2,12:19]


xline1=predicted_value[1:2, 0:6]
yline1=predicted_value[1:2, 6:12]
zline1=predicted_value[1:2, 12:19]


ax.scatter3D(xline, zline, yline,  cmap='Greens',label='Test data')
ax.scatter3D(xline1, zline1, yline1,  cmap='Reds',label='Predicted data')
ax.legend()
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.savefig('Sample1.png')
plt.show()
