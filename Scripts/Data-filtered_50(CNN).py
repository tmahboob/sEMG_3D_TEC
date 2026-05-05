from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique

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
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, BatchNormalization, Conv1D
from tensorflow.keras import backend
from tensorflow.keras.models import load_model


#Load ECG data
#The dataset you will use is based on one from timeseriesclassification.com.

# Download the dataset
dataframe = pd.read_csv('sEMG_prediction_dataset(channel_and_3Dkeypoints).csv')
raw_data = pd.DataFrame(dataframe)#dataframe.values
raw=raw_data.values



# The last element contains the labels

labels=raw[:,9:27].astype(float)
#print(labels)
#y_test = dfTest_dataValues[:, len(dfTest_data.columns) - 1]\n

#labels=labels[:,-1].values

y=labels
print(y)
# The other data points are the electrocadriogram data
data = raw[:, 1:9].astype(float)
print(data)



train_data2 = data.reshape(data.shape[0], data.shape[1], 1).astype( 'float32')
print('train_data_shape:', train_data2.shape)
#y = tf.squeeze(y, axis=-1)
#print('new y',y)


train_data, test_data, train_labels, test_labels = train_test_split( train_data2,y, test_size=0.2 )
print('train_data, test_data, train_labels, test_labels')

print(train_data, test_data, train_labels, test_labels)






#train_data, test_data, train_labels, test_labels = train_test_split( data,labels, test_size=0.2, random_state=21 )


model = Sequential()
model.add(Conv1D(64, 1, activation="relu", input_shape=(8,1),  padding='same'))
model.add(Conv1D(128, 1, activation="relu", padding='same'))
#model.add(Dense(12, activation="relu"))
model.add(Flatten())
#model.add(Dense(256, activation = 'relu'))
#model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(18))
loss = tf.keras.losses.Huber(delta=0.5)

#model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam",    metrics = ['accuracy'])
model.compile(optimizer = keras.optimizers.Adam(learning_rate=pow(10 , -4)), loss=loss)
model.summary()

model.fit(train_data, train_labels,epochs=100)

acc = model.evaluate(test_data, test_labels)
# print("Loss:", acc[0])
predictions = model.predict(test_data[:])
print("predictions shape:", predictions.shape)

from sklearn.linear_model import LogisticRegression




from sklearn.linear_model import LogisticRegression

#clf = LogisticRegression(random_state=0).fit(test_data, test_labels)
#clf.predict(test_data)

#clf.predict_proba(test_data)

#hh=clf.score(test_data, test_labels)
#print(hh)

#LR = linear_model.LinearRegression()
#pred=model.predict_proba(test_data,test_labels)
#print(pred)

pred = model.predict(test_data)
print('pred', pred)



#pred_y = pred.argmax(axis=-1)
#print('pred_y',pred_y)
#y_test=np.argmax(test_labels,axis=-1)
#print('y_test',y_test)
#cm = confusion_matrix(test_labels, pred)
#print(cm)


#y_pred=model.predict(X_test)
#y_pred=np.argmax(y_pred, axis=1)
#y_test=np.argmax(y_test, axis=1)
#cm = confusion_matrix(y_test, y_pred)

def calculate_accuracy1(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    predict = (predict)
    # diffA = real - predict
    # diff = np.absolute(real-predict)
    # diffMean = np.mean(diff)
    # diffvar = np.var(diff)
    # diffstd = np.std(diff)
    # diffmin = np.min(diff)
    # diffmax = np.max(diff)

    # percentage =  1-np.sqrt(np.mean(np.square((real - predict) ))
    threshold = 30
    rmse = np.sqrt(np.mean(np.square(real - predict)))
    print('RMSE :', rmse)
    count = 0
    for i in range(1, len(real)):
        if np.mean(np.sqrt((np.square(real[i, 1:2] - predict[i, 1:2])))) < threshold:
            count = count + 1
            if np.mean(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])))) < threshold:
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
                                            if np.mean(np.sqrt((np.square(
                                                    real[i, 10:11] - predict[i, 10:11])))) < threshold:
                                                count = count + 1
                                                if np.mean(np.sqrt((np.square(
                                                        real[i, 11:12] - predict[i, 11:12])))) < threshold:
                                                    count = count + 1
                                                    if np.mean(np.sqrt((np.square(
                                                            real[i, 12:13] - predict[i, 12:13])))) < threshold:
                                                        count = count + 1
                                                        if np.mean(np.sqrt((np.square(
                                                                real[i, 13:14] - predict[i, 13:14])))) < threshold:
                                                            count = count + 1
                                                            if np.mean(np.sqrt((np.square(
                                                                    real[i, 14:15] - predict[i, 14:15])))) < threshold:
                                                                count = count + 1
                                                                if np.mean(np.sqrt((np.square(
                                                                        real[i, 15:16] - predict[i,
                                                                                         15:16])))) < threshold:
                                                                    count = count + 1
                                                                    if np.mean(np.sqrt((np.square(
                                                                            real[i, 16:17] - predict[i,
                                                                                             16:17])))) < threshold:
                                                                        count = count + 1
                                                                        if np.mean(np.sqrt((np.square(
                                                                                real[i, 17:18] - predict[i,
                                                                                                 17:18])))) < threshold:
                                                                            count = count + 1
                                                                            if np.mean(np.sqrt((np.square(
                                                                                    real[i, 18:19] - predict[i,
                                                                                                     18:19])))) < threshold:
                                                                                count = count + 1

        # print("rmse for 1st is:", np.sqrt((np.square(real[2]-predict[2]))/6))
        print("count", count)

    # percentage2 = np.mean(np.sqrt(np.square(real - predict)))
    # percentage3 = np.mean(np.square(np.square(real - predict)))
    # import pandas as pd
    # pred_dfError = pd.DataFrame(diffA)
    # pred_dfError.to_csv('trap.csv')

    return rmse, count / (np.multiply(18, len(real)))  # )len(real) # diffMean, diffvar, diffstd, diffmin, diffmax


def calculate_accuracy(real, predict):
    threshold = 30
    rmse = np.sqrt(np.mean(np.square(real - predict)))
    print('RMSE :', rmse)
    count = 0
    for i in range(1, len(real)):
        if np.sqrt((np.square(real[i, 0:1] - predict[i, 0:1])) + (np.square(real[i, 6:7] - predict[i, 6:7])) + (
                np.square(real[i, 12:13] - predict[i, 12:13]))) < threshold:
            count = count + 1
            # print('threshold:', threshold)
            # print((np.sqrt((np.square(real[i, 0:1] - predict[i, 0:1])) + (np.square(real[i, 6:7] - predict[i, 6:7])) + (
            # np.square(real[i, 12:13] - predict[i, 12:13])))))
            # print('thumb count for', 'sample numnber: ', i, count)
        else:
            print("thumb count not updated for iteration: ", i, "and count remains", count)
        if (np.sqrt((np.square(real[i, 1:2] - predict[i, 1:2])) + (np.square(real[i, 7:8] - predict[i, 7:8])) + (
                np.square(real[i, 13:14] - predict[i, 13:14])))) < threshold:
            count = count + 1
            # print(np.sqrt((np.square(real[i, 1:2] - predict[i, 1:2])) + (np.square(real[i, 7:8] - predict[i, 7:8])) + (
            # np.square(real[i, 13:14] - predict[i, 13:14]))))
            # print('threshold:', threshold)
            # print('index count:', 'sample numnber: ', i, count)
        else:
            print("index count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
                np.square(real[i, 14:15] - predict[i, 14:15])))) < threshold:
            count = count + 1
            # print('threshold:', threshold)
            # print(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
            # np.square(real[i, 14:15] - predict[i, 14:15]))))
            # print('middle count:', 'sample numnber: ', i, count)
        else:
            print("middle count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 3:4] - predict[i, 3:4])) + (np.square(real[i, 9:10] - predict[i, 9:10])) + (
                np.square(real[i, 15:16] - predict[i, 15:16])))) < threshold:
            count = count + 1
            # print('threshold:', threshold)
            # print(np.sqrt((np.square(real[i, 2:3] - predict[i, 2:3])) + (np.square(real[i, 8:9] - predict[i, 8:9])) + (
            # np.square(real[i, 14:15] - predict[i, 14:15]))))
            # print('ring count:''sample numnber: ', i, count)
        else:
            print("ring count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5])) + (np.square(real[i, 10:11] - predict[i, 10:11])) + (
                np.square(real[i, 16:17] - predict[i, 16:17])))) < threshold:
            count = count + 1
            # print('threshold:', threshold)
            # print(np.sqrt((np.square(real[i, 4:5] - predict[i, 4:5])) + (np.square(real[i, 10:11] - predict[i, 10:11])) + (
            # np.square(real[i, 16:17] - predict[i, 16:17]))))
            # print('pinky count:', 'sample numnber: ', i, count)
        else:
            print("pinky count not updated for iteration: ", i, "and count remains", count)

        if (np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6])) + (np.square(real[i, 11:12] - predict[i, 11:12])) + (
                np.square(real[i, 17:18] - predict[i, 17:18])))) < threshold:
            count = count + 1
            # print('threshold:', threshold)
            # print(np.sqrt((np.square(real[i, 5:6] - predict[i, 5:6])) + (np.square(real[i, 11:12] - predict[i, 11:12])) + (
            # np.square(real[i, 17:18] - predict[i, 17:18]))))
            # print('palm count:', 'sample numnber: ', i, count)
        else:
            print("palm count not updated for iteration: ", i, "and count remains", count)

        # print("rmse for 1st is:", np.sqrt((np.square(real[2]-predict[2]))/6))
    print("Count Final: ", count)

    # percentage2 = np.mean(np.sqrt(np.square(real - predict)))
    # percentage3 = np.mean(np.square(np.square(real - predict)))
    # import pandas as pd
    # pred_dfError = pd.DataFrame(diffA)
    # pred_dfError.to_csv('trap.csv')
    print('length of real', len(real))
    return rmse, count / (np.multiply(6, len(real)))  # )len(real) # diffMean, diffvar, diffstd, diffmin, diffmax



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
