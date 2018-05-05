!pip install --upgrade -q gspread

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from keras import optimizers

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

worksheet = gc.open('StockDataWeek.csv').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

np.random.seed(7)
data = np.array(rows)

scl = MinMaxScaler()
#scl = StandardScaler()

data = scl.fit_transform(data)

X = data[:,0:11]
Y = data[:,11]


worksheet1 = gc.open('StockTestWeek.csv').sheet1

# get_all_values gives a list of rows.
t = worksheet1.get_all_values()

test = np.array(t)

testdata = scl.fit_transform(test)
Xtest = testdata[:,:11]
Ytest = testdata[:,11]

X = X.reshape((X.shape[0],X.shape[1],1))
Xtest = Xtest.reshape((Xtest.shape[0],Xtest.shape[1],1))

model = Sequential()
model.add(LSTM(200,input_shape=(11,1)))
#model.add(LSTM(128))
#model.add(Dense(64))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.6, nesterov=True)

model.compile(optimizer='adam',loss='mse')
model.summary()

history = model.fit(X,Y,epochs=40,validation_data=(Xtest,Ytest),shuffle=False,batch_size=7)
#history = model.fit(X,Y,epochs = 50,shuffle = False)

plt.plot(history.history['loss'])
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")

predict = model.predict(Xtest)
plt.plot(predict,label= "Predictions")
plt.plot(Ytest,label="True Values")
plt.legend()
plt.title("Predictions")
plt.xlabel("Days after training date")
plt.ylabel("Normalized Stock Closing Price")

Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1])
outArray = np.concatenate((Xtest,predict),axis=1)
val = scl.inverse_transform(outArray)

predVal = val[:,11]
test = test.astype(float)
trueVal = test[:,11]

plt.plot(predVal,label="Predicted Value")
plt.plot(trueVal, label="True Value")

print(predVal)

plt.legend()
plt.title("True Value vs Predicted Value")
plt.xlabel("Days after training data")
plt.ylabel("Google Stock Value")

plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')

print(min(history.history['val_loss']))
print(min(history.history['loss']))